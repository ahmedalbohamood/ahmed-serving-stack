#!/usr/bin/env bash
# Green-check verifier for W2D5 — Compose + secured /v1.
set -u

fail() { echo "GREEN CHECK: FAIL ($1)"; exit 1; }

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

: "${HOST_PORT:?HOST_PORT not set (check .env)}"
: "${API_KEY:?API_KEY not set (check .env)}"

BASE_URL="http://localhost:${HOST_PORT}"

echo "starting compose stack..."
docker compose up -d >/dev/null 2>&1 || fail "docker compose up failed"

echo "waiting for service to become healthy..."
deadline=$(( $(date +%s) + 180 ))
healthy=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  status=$(docker compose ps --format '{{.Health}}' serving 2>/dev/null)
  if [ "$status" = "healthy" ]; then healthy=1; break; fi
  sleep 3
done
[ "$healthy" -eq 1 ] || fail "service did not become healthy in time"

code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/health")
[ "$code" = "200" ] || fail "/health without key returned $code, expected 200"

code=$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/v1/models")
[ "$code" = "401" ] || fail "/v1/models without key returned $code, expected 401"

code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/v1/models")
[ "$code" = "200" ] || fail "/v1/models with key returned $code, expected 200"

resp=$(curl -s "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{"model":"'"${MODEL_ID}"'","messages":[{"role":"user","content":"Say hi."}],"max_tokens":16}')

echo "$resp" | grep -q '"chat.completion"' || fail "authenticated completion did not return a chat.completion"
echo "$resp" | grep -q '"content"' || fail "authenticated completion had no content field"

echo "auth: 401 without key, 200 with key"
echo "completion: ok"
echo "GREEN CHECK: PASS"
exit 0
