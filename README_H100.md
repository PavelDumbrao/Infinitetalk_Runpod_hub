# InfiniteTalk H100 Optimization Guide

## Проблема

Оригинальный образ `InfiniteTalk_Runpod_hub` не работает на H100 из-за ошибки:
```
AssertionError: SM90 kernel is not available
```

Это происходит потому, что библиотека SageAttention была собрана без поддержки архитектуры SM90 (H100 Hopper).

## Решение

Этот форк содержит два новых Dockerfile, оптимизированных для H100:
- `base_h100.Dockerfile` — базовый образ с PyTorch 2.7, CUDA 12.8 и SageAttention, собранным с поддержкой SM90
- `Dockerfile_h100` — полный образ InfiniteTalk для RunPod Serverless

## Ключевые изменения

### В base_h100.Dockerfile:
1. **PyTorch 2.7.0 с CUDA 12.8**
   ```dockerfile
   RUN pip install torch==2.7.0 torchvision==0.20.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
   ```

2. **xFormers для cu128**
   ```dockerfile
   RUN pip install xformers --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Triton 3.3.0** (необходим для SageAttention на H100)
   ```dockerfile
   RUN pip install triton==3.3.0
   ```

4. **SageAttention с SM90**
   ```dockerfile
   RUN git clone https://github.com/thu-ml/SageAttention.git && \\
       cd SageAttention && \\
       sed -i 's/# HAS_SM90 = False/HAS_SM90 = True/g' setup.py && \\
       TORCH_CUDA_ARCH_LIST="9.0" python setup.py bdist_wheel && \\
       pip install dist/*.whl
   ```

5. **Установка TORCH_CUDA_ARCH_LIST=9.0**
   ```dockerfile
   ENV TORCH_CUDA_ARCH_LIST="9.0"
   ```

## Инструкция по сборке

### Шаг 1: Соберите базовый образ

```bash
# Замените <your-dockerhub-username> на ваш Docker Hub username
docker build -t <your-dockerhub-username>/infinitetalk-h100-base:1.0 -f base_h100.Dockerfile .
docker push <your-dockerhub-username>/infinitetalk-h100-base:1.0
```

### Шаг 2: Обновите Dockerfile_h100

Откройте `Dockerfile_h100` и замените:
```dockerfile
FROM <your-dockerhub-username>/infinitetalk-h100-base:1.0 as runtime
```

на ваш реальный Docker Hub username.

### Шаг 3: Соберите полный образ

```bash
docker build -t <your-dockerhub-username>/infinitetalk-h100:1.0 -f Dockerfile_h100 .
docker push <your-dockerhub-username>/infinitetalk-h100:1.0
```

### Шаг 4: Деплой на RunPod

1. Зайдите в RunPod Console → Serverless → Endpoints
2. Создайте новый endpoint или отредактируйте существующий
3. В поле **Container Image** укажите:
   ```
   <your-dockerhub-username>/infinitetalk-h100:1.0
   ```
4. В **GPU Type** выберите:
   - `H100 PCIe` или
   - `H100 80GB PCIe`
5. В **Advanced** → **CUDA Version** выберите `12.8` или `12.9`
6. Сохраните и запустите endpoint

## Проверка работоспособности

После запуска воркера проверьте логи:
```
CUDA available: True
CUDA version: 12.8
PyTorch version: 2.7.0+cu128
```

При генерации видео не должно быть ошибок `SM90 kernel is not available`.

## Технические детали

### Почему это важно для H100

| Компонент | Оригинальная версия | H100-оптимизированная |
|-----------|--------------------|-----------------------|
| PyTorch | 2.4.1+cu121 | 2.7.0+cu128 |
| CUDA | 12.1 | 12.8 |
| SageAttention | Без SM90 | С SM90 (arch 9.0) |
| Triton | Старая версия | 3.3.0 |
| xFormers | cu121 | cu128 |

H100 (Hopper) использует compute capability **9.0** (SM90), которая требует:
- CUDA 12.3+ (рекомендуется 12.8+)
- PyTorch, собранный с поддержкой sm_90
- Все CUDA-расширения (SageAttention, Flash Attention и т.д.) должны быть скомпилированы с флагом архитектуры 9.0

## Источники проблемы

Согласно issue #1554 в ComfyUI-WanVideoWrapper и #320 в SageAttention, SM90 backend имел проблемы в некоторых версиях. Этот форк решает проблему путем явной сборки с правильными флагами.

## Альтернатива: использование A100/L40S

Если вы не хотите пересобирать образ, используйте оригинальный `InfiniteTalk_Runpod_hub` на GPU:
- A100 (SM80)
- L40S (SM89)
- RTX 4090 (SM89)

Они работают с оригинальным образом без модификаций.

## Дополнительная информация

- Оригинальный репозиторий: [wlsdml1114/Infinitetalk_Runpod_hub](https://github.com/wlsdml1114/Infinitetalk_Runpod_hub)
- InfiniteTalk: [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk)
- SageAttention: [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)

## Поддержка

Если возникают проблемы:
1. Проверьте, что в RunPod выбран H100 и CUDA 12.8+
2. Убедитесь, что базовый образ успешно собрался и загрузился в Docker Hub
3. Проверьте логи воркера на наличие ошибок сборки SageAttention
