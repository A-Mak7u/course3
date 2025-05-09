from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

# Проверим, установлен ли правильный режим
print("Current mixed precision policy:", mixed_precision.global_policy())
