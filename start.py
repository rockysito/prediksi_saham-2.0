import subprocess
import time

def main():
    print("Starting FastAPI Backend...")
    api = subprocess.Popen(["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"])
    
    print("Starting Scheduler...")
    sched = subprocess.Popen(["python", "src/pipelines/scheduler.py", "--task", "both"])
    
    time.sleep(5)
    
    print("Starting Streamlit Frontend...")
    ui = subprocess.Popen(["streamlit", "run", "frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0"])
    
    try:
        ui.wait()
    except KeyboardInterrupt:
        api.terminate()
        sched.terminate()
        ui.terminate()

if __name__ == "__main__":
    main()
