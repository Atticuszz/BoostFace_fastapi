# coding=utf-8
import time
import uuid


def main():
    start_time = time.time()
    for _ in range(1000):
        _ = uuid.uuid4()
    end_time = time.time()

    print(f"生成 1000 个 UUID 耗时: {(end_time - start_time) / 1000} 秒")


if __name__ == '__main__':
    main()

# -6级别，忽略不计
