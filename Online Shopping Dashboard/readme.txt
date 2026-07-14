================================================================
DATS6401 Final Term Project — Online Shopping Analytics Dashboard
Author  : Jeongwon Yoo
Course  : Visualization of Complex Data (DATS6401), Spring 2026
IDE     : PyCharm (Python 3.11+)
================================================================

================================================================
FILES INCLUDED
================================================================
  app.py                  -- Entry point. Run this to start the dashboard.
  dashboard.py            -- Dash layout (all tabs and UI components).
  callbacks.py            -- All interactive callbacks.
  phase1_Yoo.py           -- Phase I static plots (run separately).
  readme.txt              -- This file.
  online_shopping.csv     -- Dataset (place in same folder as app.py).

================================================================
REQUIRED PACKAGES
================================================================
Install all dependencies with:

    pip install dash dash-bootstrap-components dash-daq plotly
    pip install pandas numpy scipy scikit-learn statsmodels
    pip install matplotlib seaborn prettytable

================================================================
HOW TO RUN — Phase I (Static Plots)
================================================================
1. Open PyCharm and set the working directory to this folder.
2. Make sure online_shopping.csv is in the same folder.
3. Run:
       python term_project_updated.py
4. All static plots will be displayed sequentially.
   Tables will be printed to the console.

================================================================
HOW TO RUN — Phase II (Interactive Dashboard)
================================================================
1. Open terminal and navigate to this project folder:
       cd "/path/to/term_project_Yoo"

2. Update DATA_PATH in dashboard.py (line ~37) to match your
   local CSV file path. Example:
       DATA_PATH = '/Users/yourname/Desktop/online_shopping.csv'

3. Start the dashboard:
       python app.py

4. Open your browser and go to:
       http://127.0.0.1:8050

5. The dashboard opens with 9 interactive tabs:
       Load Data | Data Cleaning | Outlier Detection | PCA |
       Normality Test | Data Transformation |
       Numerical Plots | Categorical Plots | Statistics

================================================================
DATASET
================================================================
  Name    : Online Shopping Dataset
  Source  : Provided via course materials
  File    : online_shopping.csv
  Rows    : 52,924
  Columns : 23
  Target  : Revenue (Quantity × Avg_Price)

================================================================
NOTES
================================================================
- If you encounter a "Port already in use" error, change the
  port in app.py:
      app.run(debug=True, port=8051)

- The static plots (Phase I) use matplotlib/seaborn and will
  display as popup windows. Close each window to proceed to the
  next plot.

- All float values in the dashboard are formatted to 2 decimal
  places as required.

- Phase III (GCP deployment): Docker image and GCP URL are
  described in the final report.
================================================================
