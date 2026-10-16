resource "kubernetes_namespace" "contract_analyzer" {
  metadata {
    name = var.namespace
    labels = {
      app = "contract-analyzer"
    }
  }
}