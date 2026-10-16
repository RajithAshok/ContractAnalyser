variable "namespace" {
  description = "Kubernetes namespace for all resources"
  type        = string
  default     = "contract-analyzer"
}

variable "ollama_model" {
  description = "Ollama model to use"
  type        = string
  default     = "llama3.2:1b"
}

variable "ollama_memory_limit" {
  description = "Memory limit for Ollama pod"
  type        = string
  default     = "4Gi"
}

variable "ollama_memory_request" {
  description = "Memory request for Ollama pod"
  type        = string
  default     = "2Gi"
}

variable "chroma_memory_limit" {
  description = "Memory limit for ChromaDB pod"
  type        = string
  default     = "512Mi"
}

variable "analyzer_memory_limit" {
  description = "Memory limit for analyzer pod"
  type        = string
  default     = "1Gi"
}

variable "ollama_storage" {
  description = "Storage size for Ollama models volume"
  type        = string
  default     = "5Gi"
}

variable "chroma_storage" {
  description = "Storage size for ChromaDB volume"
  type        = string
  default     = "2Gi"
}