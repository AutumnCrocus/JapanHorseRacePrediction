
import sys
import os

# プロジェクトルートパスの追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from app import app
except ImportError as e:
    print(f"Error importing app: {e}")
    sys.exit(1)

if __name__ == "__main__":
    # GitHub Actionsでのテスト用に5001ポートで起動
    print("Starting Flask app for testing on port 5001...")
    app.run(port=5001)
