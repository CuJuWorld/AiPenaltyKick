
import time

def log_and_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"⏱️ running {func.__name__}")
        
        output = func(*args, **kwargs)
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'🏁⏱️ {func.__name__} took {round(elapsed_time, 3)} seconds')
    
        return output
    
    return wrapper
