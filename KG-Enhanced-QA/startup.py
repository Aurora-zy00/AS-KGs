import subprocess
import sys
import time
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import streamlit
        import py2neo
        import langchain_openai
        import requests
        import dotenv
        import tiktoken
        print("✅ 所有依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return False


def check_env_file():
    """Check environment variables file"""
    if not os.path.exists('.env'):
        print("❌ .env 文件不存在")
        print("请创建 .env 文件并配置以下变量:")
        print("OPENAI_API_KEY=your_api_key")
        print("OPENAI_API_BASE=https://api.deepseek.com/v1")
        print("NEO4J_URL=bolt://localhost:7687")
        print("NEO4J_USER=neo4j")
        print("NEO4J_PASSWORD=your_password")
        return False

    print("✅ .env 文件存在")
    return True


def start_backend():
    """Start backend service"""
    print("🚀 启动后端服务...")
    try:
        backend_process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        time.sleep(3)

        if backend_process.poll() is None:
            print("✅ 后端服务启动成功 (PID: {})".format(backend_process.pid))
            return backend_process
        else:
            print("❌ 后端服务启动失败")
            stdout, stderr = backend_process.communicate()
            print("错误输出:", stderr)
            return None
    except Exception as e:
        print(f"❌ 启动后端服务失败: {e}")
        return None


def start_frontend():
    """Start frontend service"""
    print("🚀 启动前端服务...")
    try:
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "frontend.py",
            "--server.port=8501"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        time.sleep(5)

        if frontend_process.poll() is None:
            print("✅ 前端服务启动成功 (PID: {})".format(frontend_process.pid))
            print("🌐 前端访问地址: http://localhost:8501")
            return frontend_process
        else:
            print("❌ 前端服务启动失败")
            stdout, stderr = frontend_process.communicate()
            print("错误输出:", stderr)
            return None
    except Exception as e:
        print(f"❌ 启动前端服务失败: {e}")
        return None


def cleanup_processes(processes):
    """Clean up processes"""
    print("\n🛑 正在关闭服务...")
    for name, process in processes.items():
        if process and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name}服务已关闭")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️ 强制关闭{name}服务")
            except Exception as e:
                print(f"❌ 关闭{name}服务失败: {e}")


def main():
    """Main function"""
    print("=" * 50)
    print("🏥 医疗空间知识图谱系统启动器")
    print("=" * 50)

    if not check_dependencies():
        return 1

    if not check_env_file():
        return 1

    processes = {}

    try:
        backend_process = start_backend()
        if not backend_process:
            return 1
        processes['后端'] = backend_process

        frontend_process = start_frontend()
        if not frontend_process:
            cleanup_processes(processes)
            return 1
        processes['前端'] = frontend_process

        print("\n" + "=" * 50)
        print("🎉 系统启动完成!")
        print("📖 后端API: http://localhost:5000")
        print("🌐 前端界面: http://localhost:8501")
        print("=" * 50)
        print("按 Ctrl+C 退出系统")

        while True:
            time.sleep(2)

            for name, process in processes.items():
                if process.poll() is not None:
                    print(f"⚠️ {name}服务意外退出")
                    cleanup_processes(processes)
                    return 1

    except KeyboardInterrupt:
        print("\n📝 接收到退出信号...")
        cleanup_processes(processes)
        print("👋 系统已退出")
        return 0
    except Exception as e:
        print(f"❌ 系统运行异常: {e}")
        cleanup_processes(processes)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)