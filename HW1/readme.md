# Foldable Smartphone Diffusion Analysis (Bass Model)
By Tatev Stepanyan

This project analyzes the diffusion of **foldable smartphones (2019–2024)** using the **Bass Diffusion Model**.  
The chosen innovation is the **Huawei MateXT Ultimate Design (2024)** tri-fold smartphone, compared to earlier **bi-fold devices**.  
The analysis uses global shipment data and forecasts adoption patterns through 2025.  

## Repository Structure

```

├── img/                    # Images used in the report
│   ├── trifold.jpg         # Example image of Huawei MateXT Ultimate Design
│   ├── trifold.jpg         # Plot generated from the script (actual vs. forecasted)
│   └── zfold.jpg           # Example image of Samsung Galazy Z Fold
│
├── data/                   # Datasets used for analysis
│   └──  data.xlsx          # Source shipment data (2019–2024)
│
├── report/                 # Report files
│   ├── report_source.Rmd   # Source RMarkdown file (no code output shown)
│   ├── report.pdf          # Final compiled report
│   └── ...
│
├── script.R                # Main R script (data prep, Bass fitting, forecasting, plotting)
├── readme.md               # This documentation file

```

## How to Run

1. Clone or download this repository.
2. Open R or RStudio in the project root.
3. Run the analysis script:
  ```r
   source("script.R")
  ```

* This will:
  * Load the dataset
  * Fit the Bass diffusion model
  * Generate forecasts
  * Produce a plot comparing actual vs. forecast shipments

4. To generate the report:
   * Knit `report/report_source.Rmd` → produces `report/report.docx`.
   * Export to PDF format via MS Word.

## Methods

* **Bass Diffusion Model** was estimated using non-linear least squares (`nls` in R).
* Parameters estimated:
  * `p` → coefficient of innovation
  * `q` → coefficient of imitation
  * `M` → market potential
* Data sources: **IDC, Statista, Gizmochina, Coolest Gadgets**.
* Forecasts are global, based on cumulative shipments and adjusted for China/global ratios in 2023–2024.

  
## Notes

* 2024 global shipment value was **calculated** using China’s share of foldables in 2023 relative to global shipments, applied to China’s 2024 figure.
* The report is **code-suppressed** (no raw R code), while full scripts are provided separately in `.R` format.
* All numeric references are from credible sources and cited in the report.

  
## Requirements

* R ≥ 4.2.0
* Packages:
  ```r
  install.packages("knitr")
  ```
