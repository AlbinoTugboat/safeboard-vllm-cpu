# Qwen на CPU в Minikube

Сервис инференса Qwen2.5-1.5B-Instruct BF16 на vLLM 0.32.0. Развёртывание описано манифестами Kubernetes и Kustomize. Модель загружается в PVC отдельным init-контейнером; для запросов используется OpenAI-совместимый API.

Чураков Н.М. · Команда исследования технологий машинного обучения

## Развёртывание

Требуются Docker Desktop с WSL2, Minikube, kubectl и Python 3.12. Команды приведены для PowerShell и выполняются из корня репозитория. На тестовой машине для WSL2 установлен лимит памяти 20 ГБ.

Первый запуск:

```powershell
docker desktop start
minikube start -p safeboard --driver=docker --container-runtime=containerd --cpus=12 --memory=16384 --kubernetes-version=v1.36.5 --listen-address=127.0.0.1
docker build -t safeboard-vllm:0.32.0 -f Dockerfile .
docker build -t safeboard-downloader:1.3 -f Dockerfile.downloader .
minikube image load safeboard-vllm:0.32.0 safeboard-downloader:1.3 -p safeboard
kubectl --context safeboard apply -f k8s/namespace.yaml
python scripts/create_secret.py --context safeboard
kubectl --context safeboard apply -k .
kubectl --context safeboard -n safeboard-llm rollout status deployment/vllm --timeout=15m
```

Утилита `create_secret.py` создаёт API-ключ в Secret `vllm-tokens` и сохраняет существующий Secret при повторном вызове. Ключ передаётся kubectl через stdin. Для публичной Qwen токен Hugging Face не требуется. При использовании закрытой модели утилита получает его из переменной окружения `HF_TOKEN`.

Первая загрузка модели занимает около 3,1 ГБ. При последующих стартах init-контейнер проверяет локальные файлы и использует кэш.

После остановки существующего профиля достаточно:

```powershell
docker desktop start
minikube start -p safeboard
```

Для применения изменённых манифестов:

```powershell
kubectl --context safeboard apply -k .
```

## Запросы к API

В отдельном терминале:

```powershell
kubectl --context safeboard -n safeboard-llm port-forward --address=127.0.0.1 service/vllm 8000:8000
```

Запрос через клиент:

```powershell
python scripts/client.py "Объясни, что такое контейнер, двумя предложениями."
```

Endpoint — `POST /v1/chat/completions`, имя модели — `safeboard-qwen`. Клиент читает API-ключ из Secret. Для работы вне локального кластера ключ можно передать через `LLM_API_TOKEN`. Клиент не использует системные HTTP-прокси и не пересылает ключ при перенаправлениях. Незашифрованный HTTP разрешён только для loopback.

## Конфигурация

| Параметр | Значение |
|---|---|
| Кластер | Kubernetes 1.36.5, 12 логических CPU, 16 ГиБ RAM |
| Основной контейнер | requests: 8 CPU / 8 ГиБ; limits: 10 CPU / 12 ГиБ |
| Init-контейнер | limits: 2 CPU / 1 ГиБ |
| Хранилище | PVC 10 ГиБ; отдельный дисковый emptyDir до 4 ГиБ |
| Модель | Qwen2.5-1.5B-Instruct, BF16, только CPU |
| Контекст и KV-кэш | 4096 токенов; 2 ГиБ |
| CPU-потоки | OMP/MKL 7, привязка `auto` |
| Батч | до 4 последовательностей и 1024 токенов |

Параметры vLLM находятся в `k8s/serve.yaml`, переменные CPU — в `k8s/runtime.env`. Kustomize создаёт ConfigMap с хешем в имени: после `kubectl apply -k .` изменение конфигурации запускает обновление Pod.

После изменения кода нужно пересобрать соответствующий образ и повторить `minikube image load`. Если тег образа сохранён, обновление запускается явно:

```powershell
kubectl --context safeboard -n safeboard-llm rollout restart deployment/vllm
kubectl --context safeboard -n safeboard-llm rollout status deployment/vllm --timeout=15m
```

Число потоков подобрано для машины с восемью физическими ядрами и SMT. Для другого CPU требуется отдельное измерение.

## Хранение модели и доступ

Ревизия модели и SHA-256 файлов закреплены в `model.lock.json`. Init-контейнер скачивает файлы через `.partial`, проверяет их и атомарно переименовывает каталог. Незавершённая загрузка сохраняется для следующей попытки. Повреждённый готовый кэш отклоняется без повторного скачивания.

Основной контейнер монтирует PVC только для чтения. Перед запуском он копирует файлы из manifest в `/verified-model/model`, проверяя размер и SHA-256 копируемых байтов. vLLM читает эту отдельную копию в offline-режиме. На каждом старте копия создаётся заново; это требует около 3,1 ГБ дополнительного диска и учитывается в файловом кэше памяти.

Оба контейнера работают с UID/GID 10001, read-only root filesystem и без Linux capabilities. Downloader получает только HF token, inference — только API-ключ. Bearer-авторизация обязательна для всех HTTP-маршрутов, кроме `GET/HEAD /health`.

В журнале запросов сохраняются ID, операция, статус и длительность. Логирование prompt и ответа vLLM отключено. Форматтер и Python exception hooks исключают текст сообщений и исключений; произвольный native/raw stdout они не фильтруют. Реализация и результаты проверок описаны в [отчёте по безопасности](results/security-fixes.md).

## Проверки

Локальные тесты и проверка работающего сервиса:

```powershell
python -m unittest discover -s tests -v
python scripts/verify_endpoint.py --context safeboard
python scripts/check_offline_cache.py
```

На Windows выполняются 40 тестов, ещё 11 требуют POSIX. Полный набор из 51 теста проверен в Linux-контейнере без сети, с UID 10001 и read-only filesystem:

```powershell
docker run --rm --network none --read-only --cap-drop ALL --security-opt no-new-privileges --tmpfs /tmp:rw,nosuid,nodev,size=64m -v "${PWD}:/review:ro" -w /review --entrypoint python safeboard-downloader:1.3 -m unittest discover -s tests -v
```

Тесты охватывают авторизацию, приватность логов, защиту API-ключа, целостность кэша, прерванную загрузку и расчёт метрик. [Контрольный запуск с пустым PVC](results/final-acceptance.json) от 23 октября включает первую загрузку модели, генерацию и повторный запуск из кэша.

## Производительность

Измерения от 23 октября: порядок A–B–B–A, по 24 запроса на сценарий и конфигурацию, 64 выходных токена. A — 8 потоков без привязки; B — 7 потоков с `auto`. Входы одинаковы, прогрев исключён, уникальный `cache_salt` исключает повторное использование префикса. В таблице приведены результаты B.

| Вход | Клиенты | TTFT среднее / p95, с | Задержка среднее / p95, с | Выходных токенов/с |
|---|---:|---:|---:|---:|
| 43 токена | 1 | 0,118 / 0,136 | 4,437 / 4,488 | 14,424 |
| 43 токена | 4 | 0,254 / 0,311 | 4,627 / 4,698 | 55,208 |
| 1717 токенов | 1 | 2,098 / 2,150 | 6,507 / 6,603 | 9,835 |
| 1717 токенов | 4 | 3,896 / 8,109 | 12,681 / 16,641 | 20,070 |

TTFT — время до первого текстового SSE-фрагмента; throughput — сумма выходных токенов, делённая на время серии. При переходе с A на B throughput вырос на 16,4–54,7% в зависимости от сценария. Бюджет батча 2048 не дал достаточного общего выигрыша; оставлено 1024.

24 октября проведено 20 минут смешанной нагрузки с четырьмя клиентами: 290 запросов, 0 ошибок, без OOM и прироста счётчика рестартов. Вход — до 3752 токенов, выход — 64–256. Максимум памяти среди 81 замера — 8,52 ГиБ из 12, включая файловый кэш. Результат относится к локальной закрытой нагрузке и не описывает работу сервиса за часы или дни.

Повторение измерений при работающем port-forward:

```powershell
python scripts/benchmark_decode.py --requests 24 --output-tokens 64 --prompts short long --concurrency 1 4 --kube-context safeboard --output results/current-benchmark.json
python scripts/soak_test.py --duration-seconds 1200 --concurrency 4 --kube-context safeboard --output results/current-soak.json
```

Методика, p95, сравнение конфигураций и наблюдения при перезапуске — в [результатах измерений](results/performance-2026-10-23/README.md). Общий технический отчёт — [REPORT.md](REPORT.md).

## Обслуживание

```powershell
kubectl --context safeboard -n safeboard-llm get pods,pvc
kubectl --context safeboard -n safeboard-llm logs deployment/vllm -c download-model
kubectl --context safeboard -n safeboard-llm logs deployment/vllm -c inference
minikube stop -p safeboard
```

Остановка профиля сохраняет PVC. Для восстановления повреждённого кэша следует остановить Deployment и перенести повреждённый каталог модели в карантин через служебный Pod с UID/GID 10001. После возобновления Deployment загрузчик создаст каталог заново. Перед загрузкой требуется место для обеих копий.

На тестовой машине Docker Desktop потребовалась настройка CoreDNS. Если init-контейнер не разрешает имена Hugging Face, для локального одноузлового профиля `safeboard`:

```powershell
.\scripts\repair_minikube_dns.ps1
kubectl --context safeboard -n safeboard-llm rollout restart deployment/vllm
```

Утилита переводит CoreDNS в сеть узла с `dnsPolicy: Default`. В кластере с работающим DNS эта настройка не требуется.

## Ограничения

Сервис рассчитан на локальный доступ через port-forward. Для внешнего доступа нужны TLS и ограничения доступа и нагрузки. Известные замечания к setuptools, его встроенным jaraco.context/wheel и пакетам ОС перечислены в [results/security-fixes.md](results/security-fixes.md).

Базовые образы закреплены digest, Python-зависимости — версиями и хешами. Репозитории apt изменяемы, поэтому сборка не является побитово воспроизводимой.
