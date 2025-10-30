import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Now import and run
from app import process_analysis_sync

if __name__ == '__main__':
    # Read arguments
    session_id = sys.argv[1]
    front_path = sys.argv[2]
    side_path = sys.argv[3]
    back_path = sys.argv[4]
    height = float(sys.argv[5])
    weight = float(sys.argv[6])
    age = int(sys.argv[7])
    gender = sys.argv[8]
    
    # Run analysis
    result = process_analysis_sync(
        session_id, front_path, side_path, back_path,
        height, weight, age, gender
    )
    
    # Print result as JSON
    print(json.dumps(result))
    
    # Force exit to avoid cleanup crash
    sys.exit(0)