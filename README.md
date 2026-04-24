# Greek RES Forecasting model
## Methodology

### Phase 1 - Data collection
Pulling data from two public free APIs:
  - **ENTSO-e Transparency Platform**: hourly wind & solar generation for Greece, 2024-2025
  - **Open-meteo**: weather data (irradiance, wind speed & temperature)

### Phase 2 - Data analysis
  - Seasonal and intraday generation patterns for solar and wind technologies
  - Ramp rate analysis
  - Correlation matrix
  - Duck curve for Greece electric grid

### Phase 3 - Day-ahead Forecasting Model
Two distinct periods:
01/2024 - 09/2025 --> training period
10/2025 - 12/2025 --> forecasting/testing period

*Three models to evaluate solar and wind forecasts:*

Model | Solar Mean Error | Wind Mean Error
- Seasonal Naive | 539 MW | 1.061 MW
- Linear regression | 345 MW | 391 MW
- Random Forest | 261 MW | 427 MW

**Key findings:**
- Seasonal Naive (my baseline model) was completey off; expected
- Linear regression was a significant improvement for both technologies
- Random Forest improved **Solar** a bit further (total of 48% compared to Baseline) but received worst result for **Wind**

### Phase 4 - DAM Price Correlation & Merit order analysis
Day Ahead Prices were pulled from ENTSO-e and not HENEX. They were merged with generation data to quantify the merit order effect
- Increased RES production supresses prices by 57%
- 126 hours of negative prices, concentrated around August-September, when solar generation peaks

## Dashboard
*Interactive dashboard with the following options:*
- **Overview:** time series and monthly summary table
- **Forecast vs Actual:** Day ahead forecasts vs actual generation, error distribution
- **Merit order analysis:** RES vs price scatter plot, intraday profile
- **Negative prices:** Hourly and monthly breakdown of negative prices
- **Model performance:** comparison of MAE/RMSE across the three models
- *TBA:* 6th option, showing forecast analysis for the n+1 day

## Tech
- **Data collection:** entsoe-py, requests
- **Data processing:** pandas, numpy
- **ML:** scikit-learn (Linear regression, Random Forest)
- **Visualization:** matplotlib, seaborn, plotly
- **Dashboard:** Streamlit
- **Data sources:** Transparency Platform (ENTSO-e), Open-meteo

## How to run locally
**1. Clone the repository**
git clone https://github.com/Taxiarchis135/greek-res-forecasting.git
cd greek-res-forecasting

**2. Install dependencies**
pip install -r requirements.txt

**3. Set your ENTSO-E API key**

Register for free at [transparency.entsoe.eu](https://transparency.entsoe.eu) and set your token *(you have to send them an email with the headline "Restful API access", asking to enable the option for your account to generate a token)*:
export ENTSOE_API_KEY="your_token_here"

**4. Run data collection**
python phase1_data_collection.py
python phase1b_wind_data.py
python phase4a_price_collection.py

**5. Launch the dashboard**
streamlit run dashboard.py

## Key Insights
- A forecast error of 507MW for total RES generation represents direct imbalance exposure.
- Greek duck curve when solar generation peaks, dropping prices to ~55 €/MWh and spike to ~183 €/MWh during evening hours. Intraday spread of ~128 €/MWh, highliting the value of flexible assets (storage and DR)
- Further installation of RES capacity without the necessary storage capacity and flexible assets will lead to further price cannibalization, endangering further investments for energy transition

*This project is for portfolio and educational purposes. All data is sourced from publicly available platforms.*
