#!/bin/bash

#SBATCH -A MST114558                            ##錢包 id（已設為我們計畫的錢包）【不需修改】
#SBATCH --job-name=<job_name>                   ##工作名稱
#SBATCH --output=./execution/output_%j.log      ##標準[輸出]與[錯誤]輸出同時記錄到此檔案 ; %x for job name
#SBATCH --error=./execution/error.err           ##標準[錯誤]輸出紀錄同一檔案            ; %j for job ID
#SBATCH --nodes=1                               ##請求 n 個 節點
#SBATCH --ntasks-per-node=1                     ##每個 node 可執行 n 個任務
#SBATCH --gres=gpu:1                            ##請求 n 張 GPU（針對每個節點）
#SBATCH --cpus-per-task=4                       ##但個任務請求 n 個 CPU
#SBATCH --partition=normal                      ##測試分區：dev(H100, 2小時)、normal(H100, 48小時)、4nodes(H100, 12小時)、normal2(H200, 48小時)
#SBATCH --mail-type=ALL                         ##NONE, BEGIN, END, FAIL, ALL【不需修改】
#SBATCH --mail-user=<email>                     ##將 job 執行的狀態寄到您的 email，不需要反覆進入節點檢查

#====== 執行主程式嗎 ======
# export HF_HOME="./cache" #按照需求

#啟動 environment （如果執行時，沒有啟動）
# ml load miniconda3/24.11.1
# CONDA_DEFAULT_ENV="<environment_name>"
# conda activate "$CONDA_DEFAULT_ENV"

echo "======================"
echo "開始執行主程式碼"
echo "======================

"

#主程式碼
# python <file_name>.py <args>

sleep 10
echo "======================"
echo "執行完整！"