### AI-shared

Запуск модели vLLM (OpenAI-совместимый API на `/v1`, порт 8000).

Перед запуском задать `VLLM_API_KEY` в `.env` (см. `.env.example`) — им защищён доступ к vLLM,
он же передаётся как `Authorization: Bearer <VLLM_API_KEY>` клиентами из других проектов.

```shell
docker compose up -d
```

