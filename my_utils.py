from contextlib import contextmanager
import time

@contextmanager
def timer(code_part = "No part name"):
    start = time.perf_counter()
    yield  # Everything in the `with` block executes here
    end = time.perf_counter()
    take_time = (end - start)*1000
    print(f"***This {code_part} took about {take_time:.2f} millisecond***")
    