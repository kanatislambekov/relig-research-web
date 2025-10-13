# Kazakhstan Fertility & Religiosity Explorer

This Streamlit application presents the main quantitative findings from the fertility and religiosity study. Use the multi-page navigation to move between descriptive statistics, Poisson models, Cox hazard models, and the extended abstract accompanying the paper.

## Getting started

```bash
pip install -r requirements.txt
streamlit run Home.py
```

The app reads the Excel and DOCX artefacts located in the `Results/` folder. If you rename or replace these files, update the filenames referenced in `data_loader.py` accordingly.

To explore the new visual analytics page, add the survey microdata as `data/Kaz_Ggs.*` (CSV, Excel, SPSS, Parquet, or Feather). The loader automatically detects the extension and powers the age, fertility, and values dashboards.
