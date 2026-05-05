"""
Contract Analyzer CLI — rich terminal interface.

Usage:
    python -m src.cli                           # interactive REPL
    python -m src.cli analyze path/to/file.txt  # one-shot analysis
"""
from __future__ import annotations
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich import box

from src.models import ExtractionResult, ComparisonResult, QAResult, RiskLevel
from src.agents.router import route
from src.rag.chunker import chunk_contract
from src.rag.retriever import index_chunks, list_documents, delete_document, get_full_document_text

app = typer.Typer(help="Local AI contract analyzer powered by Ollama")
console = Console()

RISK_COLORS = {RiskLevel.HIGH: "red", RiskLevel.MEDIUM: "yellow", RiskLevel.LOW: "green"}
RISK_ICONS = {RiskLevel.HIGH: "●", RiskLevel.MEDIUM: "◐", RiskLevel.LOW: "○"}

# In-memory cache of ExtractionResults for the session
_cache: dict[str, ExtractionResult] = {}
_active_doc: str | None = None
_active_doc_b: str | None = None


def _risk_badge(level: RiskLevel, score: float | int | None = None) -> Text:
    color = RISK_COLORS[level]
    icon = RISK_ICONS[level]
    label = f"{icon} {level.value}"
    if score is not None:
        label += f"  {score}/10"
    return Text(label, style=color)


def _load_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        console.print(f"[red]File not found:[/red] {path}")
        raise typer.Exit(1)
    if p.suffix.lower() == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(p) as pdf:
                return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            console.print("[yellow]pdfplumber not installed — reading raw bytes as text[/yellow]")
    return p.read_text(errors="replace")


def _index_document(file_path: str) -> str:
    """Load, chunk, embed, and index a contract. Returns doc_name."""
    text = _load_file(file_path)
    doc_name = Path(file_path).stem
    with console.status(f"[dim]Chunking and indexing {doc_name}...[/dim]"):
        chunks = chunk_contract(text, doc_name)
        index_chunks(chunks)
    console.print(f"[green]✓[/green] Indexed [bold]{doc_name}[/bold] ({len(chunks)} chunks)")
    return doc_name


def _print_extraction(result: ExtractionResult) -> None:
    console.print()
    console.print(Panel(result.summary, title=f"[bold]{result.document_name}[/bold]", expand=False))
    console.print(f"  Overall risk: {_risk_badge(result.overall_risk, result.overall_risk_score)}")
    console.print()

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("Clause type", style="bold", width=20)
    table.add_column("Text", max_width=55)
    table.add_column("Risk", width=14)
    table.add_column("Note", max_width=35, style="dim")

    for c in sorted(result.clauses, key=lambda x: x.risk_score, reverse=True):
        color = RISK_COLORS[c.risk_level]
        table.add_row(
            c.clause_type,
            c.text[:200] + ("…" if len(c.text) > 200 else ""),
            Text(f"{RISK_ICONS[c.risk_level]} {c.risk_score}/10", style=color),
            c.risk_reason or "",
        )
    console.print(table)


def _print_risk_report(report: dict) -> None:
    counts = report["counts"]
    console.print()
    console.print(Panel(
        f"[red]High risk:[/red] {counts['high']}   "
        f"[yellow]Medium:[/yellow] {counts['medium']}   "
        f"[green]Low:[/green] {counts['low']}   "
        f"Total clauses: {counts['total']}\n\n"
        + report["summary"],
        title=f"[bold]Risk Report — {report['document_name']}[/bold]",
        expand=False,
    ))

    if report["recommendations"]:
        console.print("\n[bold]Recommendations:[/bold]")
        for i, rec in enumerate(report["recommendations"], 1):
            console.print(f"  [cyan]{i}.[/cyan] {rec}")

    if report["high_risk_clauses"]:
        console.print("\n[bold red]High-risk clauses:[/bold red]")
        for c in report["high_risk_clauses"]:
            console.print(f"  [red]●[/red] [bold]{c.clause_type}[/bold]  ({c.section_ref or '—'})")
            console.print(f"    {c.text[:180]}")
            if c.risk_reason:
                console.print(f"    [dim italic]{c.risk_reason}[/dim italic]")
            console.print()


def _print_comparison(result: ComparisonResult) -> None:
    console.print()
    console.print(Panel(result.recommendation, title="[bold]Comparison — Overall Recommendation[/bold]", expand=False))
    console.print()

    table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False)
    table.add_column("Clause", style="bold", width=18)
    table.add_column(result.doc_a_name[:24], max_width=28)
    table.add_column(result.doc_b_name[:24], max_width=28)
    table.add_column("Better", width=10)
    table.add_column("Reason", max_width=30, style="dim")

    better_map = {"doc_a": f"[green]{result.doc_a_name[:12]}[/green]",
                  "doc_b": f"[green]{result.doc_b_name[:12]}[/green]",
                  "equal": "[dim]equal[/dim]"}

    for d in result.diffs:
        table.add_row(
            d.clause_type,
            d.doc_a_text or "[dim]absent[/dim]",
            d.doc_b_text or "[dim]absent[/dim]",
            better_map.get(d.which_is_better, "—"),
            d.reason,
        )
    console.print(table)


def _print_qa(result: QAResult) -> None:
    console.print()
    console.print(Panel(result.answer, title="[bold]Answer[/bold]", expand=False))
    if result.citations:
        console.print("[dim]Citations:[/dim]")
        for c in result.citations:
            console.print(f'  [dim italic]"{c}"[/dim italic]')
    confidence_color = "green" if result.confidence > 0.7 else "yellow" if result.confidence > 0.4 else "red"
    console.print(f"\n  [dim]Confidence:[/dim] [{confidence_color}]{result.confidence:.0%}[/{confidence_color}]")


# ── CLI Commands ───────────────────────────────────────────────────────────────

@app.command()
def analyze(
    file: str = typer.Argument(..., help="Path to a .txt or .pdf contract"),
    risk: bool = typer.Option(False, "--risk", "-r", help="Also run risk analysis"),
):
    """Load and analyze a single contract file."""
    global _active_doc, _cache
    doc_name = _index_document(file)
    _active_doc = doc_name

    text = get_full_document_text(doc_name)
    from src.agents.extractor import extract_clauses
    with console.status("[dim]Extracting clauses...[/dim]"):
        result = extract_clauses(text, doc_name)
    _cache[doc_name] = result
    _print_extraction(result)

    if risk:
        from src.agents.risk_scorer import build_risk_report
        report = build_risk_report(result)
        _print_risk_report(report)


@app.command()
def repl():
    """Start the interactive REPL."""
    _run_repl()


def _run_repl():
    global _active_doc, _active_doc_b, _cache

    console.print(Panel(
        "[bold]Contract Analyzer[/bold]  •  powered by Ollama\n\n"
        "Commands: [cyan]load <file>[/cyan]  [cyan]list[/cyan]  [cyan]use <docname>[/cyan]  "
        "[cyan]extract[/cyan]  [cyan]risk[/cyan]  [cyan]compare <docname>[/cyan]  [cyan]ask <question>[/cyan]  [cyan]exit[/cyan]",
        expand=False,
    ))

    # Show already-indexed docs
    indexed = list_documents()
    if indexed:
        console.print(f"[dim]Previously indexed:[/dim] {', '.join(indexed)}")
        if not _active_doc:
            _active_doc = indexed[0]
            console.print(f"[dim]Active document:[/dim] {_active_doc}")

    while True:
        try:
            raw = Prompt.ask("\n[bold cyan]▶[/bold cyan]", default="").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        # ── load ──────────────────────────────────────────────────────
        if cmd == "load":
            if not rest:
                console.print("[yellow]Usage:[/yellow] load <file_path>")
                continue
            try:
                doc_name = _index_document(rest.strip())
                _active_doc = doc_name
            except SystemExit:
                pass

        # ── list ──────────────────────────────────────────────────────
        elif cmd == "list":
            docs = list_documents()
            if docs:
                for d in docs:
                    marker = " [cyan]← active[/cyan]" if d == _active_doc else ""
                    console.print(f"  • {d}{marker}")
            else:
                console.print("[dim]No documents indexed yet.[/dim]")

        # ── use ───────────────────────────────────────────────────────
        elif cmd == "use":
            if not rest:
                console.print("[yellow]Usage:[/yellow] use <doc_name>")
                continue
            _active_doc = rest.strip()
            console.print(f"[green]Active document:[/green] {_active_doc}")

        # ── delete ────────────────────────────────────────────────────
        elif cmd == "delete":
            target = rest.strip() or _active_doc
            if target:
                n = delete_document(target)
                _cache.pop(target, None)
                console.print(f"[dim]Deleted {n} chunks for {target}[/dim]")
            else:
                console.print("[yellow]No document specified.[/yellow]")

        # ── extract ───────────────────────────────────────────────────
        elif cmd == "extract":
            if not _active_doc:
                console.print("[yellow]Load a document first.[/yellow]")
                continue
            _ensure_extracted(_active_doc)
            _print_extraction(_cache[_active_doc])

        # ── risk ──────────────────────────────────────────────────────
        elif cmd == "risk":
            if not _active_doc:
                console.print("[yellow]Load a document first.[/yellow]")
                continue
            _ensure_extracted(_active_doc)
            from src.agents.risk_scorer import build_risk_report
            report = build_risk_report(_cache[_active_doc])
            _print_risk_report(report)

        # ── compare ───────────────────────────────────────────────────
        elif cmd == "compare":
            if not rest:
                console.print("[yellow]Usage:[/yellow] compare <doc_name_or_file>")
                continue
            target = rest.strip()
            # Accept a file path or a doc_name
            if Path(target).exists():
                target = _index_document(target)
            _active_doc_b = target
            if not _active_doc:
                console.print("[yellow]Load a primary document first.[/yellow]")
                continue
            _ensure_extracted(_active_doc)
            _ensure_extracted(_active_doc_b)
            from src.agents.comparator import compare_documents
            with console.status("[dim]Comparing documents...[/dim]"):
                result = compare_documents(_cache[_active_doc], _cache[_active_doc_b])
            _print_comparison(result)

        # ── ask ───────────────────────────────────────────────────────
        elif cmd in ("ask", "q"):
            question = rest.strip()
            if not question:
                console.print("[yellow]Usage:[/yellow] ask <your question>")
                continue
            if not _active_doc:
                console.print("[yellow]Load a document first.[/yellow]")
                continue
            with console.status("[dim]Searching contract...[/dim]"):
                from src.agents.qa import answer_question
                result = answer_question(question, _active_doc)
            _print_qa(result)

        # ── exit ──────────────────────────────────────────────────────
        elif cmd in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        # ── help ──────────────────────────────────────────────────────
        elif cmd in ("help", "?"):
            console.print(
                "  [cyan]load <file>[/cyan]         — index a .txt or .pdf contract\n"
                "  [cyan]list[/cyan]                — show all indexed contracts\n"
                "  [cyan]use <docname>[/cyan]        — switch active document\n"
                "  [cyan]extract[/cyan]             — extract and list all clauses\n"
                "  [cyan]risk[/cyan]                — run full risk analysis\n"
                "  [cyan]compare <doc>[/cyan]        — compare active doc vs another\n"
                "  [cyan]ask <question>[/cyan]       — RAG Q&A over the contract\n"
                "  [cyan]delete [doc][/cyan]         — remove document from index\n"
                "  [cyan]exit[/cyan]                — quit"
            )

        else:
            # Treat unknown input as a question if a doc is loaded
            if _active_doc:
                with console.status("[dim]Searching contract...[/dim]"):
                    from src.agents.qa import answer_question
                    result = answer_question(raw, _active_doc)
                _print_qa(result)
            else:
                console.print(f"[dim]Unknown command '{cmd}'. Type 'help' for usage.[/dim]")


def _ensure_extracted(doc_name: str) -> None:
    """Extract clauses for doc_name if not already cached."""
    if doc_name not in _cache:
        text = get_full_document_text(doc_name)
        if not text:
            console.print(f"[red]No indexed content for {doc_name}.[/red]")
            return
        from src.agents.extractor import extract_clauses
        with console.status(f"[dim]Extracting clauses from {doc_name}...[/dim]"):
            _cache[doc_name] = extract_clauses(text, doc_name)


def main():
    if len(sys.argv) > 1:
        app()
    else:
        _run_repl()


if __name__ == "__main__":
    main()
