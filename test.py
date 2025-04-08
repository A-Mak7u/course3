import psutil

# Получаем температуру процессора
cpu_temp = psutil.sensors_temperatures()

if 'coretemp' in cpu_temp:
    for entry in cpu_temp['coretemp']:
        print(f"Температура {entry.label}: {entry.current}°C")
else:
    print("Температура процессора недоступна.")
