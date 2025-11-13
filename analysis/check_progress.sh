#!/bin/bash
# Quick script to check data generation progress

echo "========================================================================"
echo "H-Net SMILES Tokenization Analysis - Progress Check"
echo "========================================================================"
echo ""

# Check if process is running
PID=$(pgrep -f "run_data_generation.py")
if [ -z "$PID" ]; then
    echo "❌ Data generation is NOT running"
    echo ""
    echo "To restart:"
    echo "  cd /home/ec2-user/hnet_smiles"
    echo "  source /opt/pytorch/bin/activate"
    echo "  nohup python3 -u analysis/run_data_generation.py > analysis/logs/data_generation_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
else
    echo "✅ Data generation is RUNNING (PID: $PID)"
    echo ""
fi

# Find latest log file
LOGFILE=$(ls -t /home/ec2-user/hnet_smiles/analysis/logs/data_generation_*.log 2>/dev/null | head -1)

if [ -z "$LOGFILE" ]; then
    echo "⚠️  No log file found"
    exit 1
fi

echo "Log file: $LOGFILE"
echo ""

# Show current progress (last 30 lines)
echo "========================================================================"
echo "Latest Progress (last 30 lines):"
echo "========================================================================"
tail -30 "$LOGFILE"

echo ""
echo "========================================================================"
echo "GPU Status:"
echo "========================================================================"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s | Utilization: %s%% | Memory: %s / %s MB\n", $1, $2, $3, $4, $5}'

echo ""
echo "========================================================================"
echo "Generated Files:"
echo "========================================================================"

# Count H-Net results
HNET_COUNT=$(ls /home/ec2-user/hnet_smiles/analysis/data/hnet_results/*.pkl 2>/dev/null | wc -l)
echo "H-Net Results: $HNET_COUNT / 6"

# Count SmilesPE results
SPE_COUNT=$(ls /home/ec2-user/hnet_smiles/analysis/data/smilesPE_results/*.pkl 2>/dev/null | wc -l)
echo "SmilesPE Results: $SPE_COUNT / 2"

# Count statistics
STATS_COUNT=$(ls /home/ec2-user/hnet_smiles/analysis/data/statistics/*.json 2>/dev/null | wc -l)
echo "Statistics: $STATS_COUNT / 8"

echo ""
echo "To monitor in real-time:"
echo "  tail -f $LOGFILE"
echo ""
echo "To check GPU:"
echo "  watch -n 1 nvidia-smi"
echo ""

