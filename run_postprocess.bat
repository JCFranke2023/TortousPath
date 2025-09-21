@echo off
REM Post-processing workflow for PECVD simulations

echo === STEP 1: Extract data from ODB files ===
REM Run with ABAQUS Python
abaqus python odb_extractor.py --batch job_list.txt --output extracted_data

echo.
echo === STEP 2: Analyze data and create plots ===
REM Run with system Python (with pandas/matplotlib)
python data_analyzer.py --sweep --data_dir extracted_data --output analysis_results

echo.
echo === COMPLETE ===
echo Results saved in analysis_results/
pause