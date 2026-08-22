# Caricare il modello fuso con `transformers`

Nota a margine per chi userà il modello fuso (`trainmaster-export`) con
`transformers`/`AutoModelForCausalLM`: va caricato con `AutoModelForImageTextToText`
(architettura `Qwen3_5ForConditionalGeneration`), non `AutoModelForCausalLM` — l'ho
verificato e annotato solo qui in chat, non serve documentarlo nel progetto perché è un
dettaglio dell'API `transformers`, non del comportamento di `trainmaster-export`.
