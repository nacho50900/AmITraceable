#!/bin/bash
# Uso: MISTRAL_API_KEY=tu_key ./test_mistral.sh
# Dispara 3 llamadas seguidas (sin esperar nada entre ellas) para ver
# exactamente qué responde Mistral -- status code + headers de rate limit
# + cuerpo completo. Sirve para confirmar si el problema es de verdad la
# API key/cuenta, y no algo del backend.

if [ -z "$MISTRAL_API_KEY" ]; then
  echo "Falta MISTRAL_API_KEY en el entorno. Uso: MISTRAL_API_KEY=xxx ./test_mistral.sh"
  exit 1
fi

for i in 1 2 3; do
  echo "=== Llamada $i ==="
  curl -s -i -X POST "https://api.mistral.ai/v1/chat/completions" \
    -H "Authorization: Bearer $MISTRAL_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
      "model": "mistral-small-latest",
      "messages": [{"role": "user", "content": "di solo la palabra hola"}],
      "max_tokens": 10
    }'
  echo
  echo
done
