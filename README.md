## Methodology

### Step 1: Pair Selection & Statistical Checks
Before applying the Kalman filter, we verify that the chosen stocks form a valid pair:

1. Unit root tests (ADF, KPSS): ensure both series are $\mathbf{I(1)}$.  
2. Engle-Granger cointegration test: ensures the residuals are stationary.  
3. Half-life estimation: checks how quickly the spread mean-reverts.  
4. Rolling betas: test for hedge ratio stability.  

Formally, an OLS regression is first run:

$\mathbf{y_{1t} = \alpha + \beta y_{2t} + \varepsilon_t}$

and we test whether the residuals $\mathbf{\varepsilon_t}$ are stationary.

Example diagnostics (from one stock pair):

$\mathbf{\hat{\beta} = -0.411, 
\quad 
\hat{\alpha} = 4.196}$

- **KPSS test: confirms one series is $\mathbf{I(1)}$ ($\mathbf{p < 0.05}$)  
- **Engle-Granger p-value: $\mathbf{0.376}$ (weak evidence of cointegration)  
- **ADF on residuals: $\mathbf{p = 0.021}$ (residuals stationary at 5\% level)  
- **Half-life: $\mathbf{\approx 57.5}$ days  
- **Rolling beta standard deviation: $\mathbf{0.291}$ (unstable hedge ratio)  

Verdict summary:  

$\mathbf{\{ y_\text{level\_nonstationary}: \text{True}, \; x_\text{level\_nonstationary}: \text{False}, \; \varepsilon_\text{stationary}: \text{False}, \; \text{half-life reasonable}: \text{True}, \; \beta_\text{stable}: \text{False} \}}$

---

### Step 2: From Linear Regression to Dynamic Parameters
The spread of the log prices of two stocks is defined as:

$\mathbf{s_t = y_{1t} - **\gamma y_{2t} = \mu + \varepsilon_t}$

where $\mathbf{\gamma}$ allows the spread to be centered around $\mathbf{\mu}$.

Then,

$\mathbf{y_{1t} = \mu + \gamma y_{2t} + \varepsilon_t}$

Assuming constant $\mathbf{\mu}$ and $\mathbf{\gamma}$ is restrictive since market relationships change over time. To capture this, the coefficients are allowed to vary dynamically:

$\mathbf{y_{1t} = \mu_t + \gamma_t y_{2t} + \varepsilon_t}$

so that at each time step $\mathbf{t}$, the spread

$\mathbf{s_t = y_{1t} - **\mu_t - **\gamma_t y_{2t}}$

remains stationary and mean-reverting around zero.

---

### Step 3: Kalman Filter Mechanics
At each time step, the Kalman filter runs two main steps: prediction and update.

#### 1. Predict

$\mathbf{\hat{\mathbf{x}}_{t}^{-} = \mathbf{A} \hat{\mathbf{x}}_{t-1}, 
\qquad
\mathbf{P}_{t}^{-} = \mathbf{A} \mathbf{P}_{t-1} \mathbf{A}^\top + \mathbf{Q}}$

- **$\mathbf{\hat{\mathbf{x}}_{t}^{-}}$ : predicted state vector (prior) at time $\mathbf{t}$  
- **$\mathbf{\hat{\mathbf{x}}_{t-1}}$ : updated state from previous step  
- **$\mathbf{A}$ : state transition matrix (identity for random walk)  
- **$\mathbf{P}_{t}^{-}$ : predicted error covariance  
- **$\mathbf{P}_{t-1}$ : updated error covariance from previous step  
- **$\mathbf{Q}$ : process noise covariance  

> *Predicts the next state and its uncertainty before seeing the observation.*

#### 2. Compute Kalman Gain & Update

$\mathbf{\mathbf{K}_t = \mathbf{P}_{t}^{-} \mathbf{H}_t^\top (\mathbf{H}_t \mathbf{P}_{t}^{-} \mathbf{H}_t^\top + \mathbf{R})^{-1}}$

$\mathbf{\hat{\mathbf{x}}_{t} = \hat{\mathbf{x}}_{t}^{-} + \mathbf{K}_t (\mathbf{z}_t - **\mathbf{H}_t \hat{\mathbf{x}}_{t}^{-})}$

$\mathbf{\mathbf{P}_{t} = (\mathbf{I} - **\mathbf{K}_t \mathbf{H}_t) \mathbf{P}_{t}^{-}}$

- **$\mathbf{K}_t$ : Kalman gain, determines weighting of observation vs. prior  
- **$\mathbf{H}_t$ : observation matrix, here $\mathbf{[1, y_{2t}]}$  
- **$\mathbf{R}$ : observation noise covariance  
- **$\mathbf{z}_t$ : observed measurement ($\mathbf{y_{1t}}$)  
- **$\hat{\mathbf{x}}_t$ : updated state estimate (posterior)  
- **$\mathbf{P}_t$ : updated error covariance  

> *Updates the predicted state using the new observation.*

#### 3. Compute Spread

$\mathbf{s_t = y_{1t} - **\mu_t - **\gamma_t y_{2t}}$

- **$\mathbf{s_t}$ : residual spread at time $\mathbf{t}$  
- **$\mathbf{\mu_t, \gamma_t}$ : dynamically estimated intercept and hedge ratio  

> *The spread $\mathbf{s_t}$ is expected to mean-revert around zero.*

---

### Step 4: Implementation
The Kalman filter is implemented in `pairs_trading_kalman.ipynb` as a custom Python class estimating dynamic $\mathbf{\mu_t}$ and $\mathbf{\gamma_t}$.

#### Initialization

$\mathbf{x_0 =
\begin{bmatrix}
0 \\
1
\end{bmatrix}, 
\qquad
P_0 = I}$

- **$\mathbf{x_0}$ : initial state vector $\mathbf{[\mu_0, \gamma_0]^\top}$  
- **$\mathbf{P_0}$ : initial covariance matrix  
- **$\mathbf{I}$ : $\mathbf{2\times2}$ identity matrix  

> *Assumes neutral intercept and unit hedge ratio at $\mathbf{t=0}$.*

#### Noise Terms

- **Process noise $\mathbf{Q}$: represents uncertainty in $\mathbf{\mu_t}$ and $\mathbf{\gamma_t}$  
- **Observation noise $\mathbf{R}$: rolling variance of $\mathbf{y_1}$  

#### Dynamic Estimation

1. Predict:

$\mathbf{x_t^- **= A x_{t-1}, 
\quad
P_t^- **= A P_{t-1} A^\top + Q}$

2. Update:

$\mathbf{K_t = P_t^- **H_t^\top (H_t P_t^- **H_t^\top + R)^{-1}}$

$\mathbf{x_t = x_t^- **+ K_t (z_t - **H_t x_t^-)}$

$\mathbf{P_t = (I - **K_t H_t) P_t^-}$

- **$\mathbf{x_t = [\mu_t, \gamma_t]^\top}$ : updated estimates  
- **$\mathbf{H_t = [1, y_{2t}]}$ : observation matrix  
- **$\mathbf{z_t = y_{1t}}$ : observed log-price  
- **$\mathbf{K_t}$ : Kalman gain  

> *Filter adjusts $\mathbf{\mu_t}$ and $\mathbf{\gamma_t}$ to best fit new observations.*

#### Compute Spread

$\mathbf{s_t = y_{1t} - **\mu_t - **\gamma_t y_{2t}}$

> *Updated $\mathbf{\mu_t}$ and $\mathbf{\gamma_t}$ are used to compute the spread.*

#### Outputs

- **Time series of $\mathbf{\mu_t}$ (intercept)  
- **Time series of $\mathbf{\gamma_t}$ (hedge ratio)  
- **Spread $\mathbf{s_t}$  

![Filtered Plot](results/mu_gamma.png)
![Filtered Plot](results/Stationary.png)

---

### Step 5: QuantConnect Implementation
The Kalman filter-based pairs trading strategy is implemented in QuantConnect, trading $\mathbf{y_1}$ and $\mathbf{y_2}$ dynamically based on the estimated spread and hedge ratio.

#### Trading Logic

- **Z-score of spread:

$\mathbf{z_t = \frac{s_t - **\text{mean}(s_{t-w:t})}{\text{std}(s_{t-w:t})}}$

- **Entry Rules: Initiate positions when the spread deviates significantly from its mean. Use $\mathbf{\gamma_t}$ for market-neutral sizing:

$\mathbf{\text{hedge shares} = \text{base shares} \cdot \gamma_t \cdot \frac{y_1}{y_2}}$

- **Exit Rules: Close positions when spread reverts or losses exceed risk thresholds.

- **Realized P&L:

$\mathbf{\text{PNL}_t = (y_{1t} - **y_{1,\text{entry}}) \cdot q_1 + (y_{2t} - **y_{2,\text{entry}}) \cdot q_2}$

#### Outputs & Visualization

![Filtered Plot](results/QC_Graph.png)
![Filtered Plot](results/QC_Results.png)

---

## Future Work
- **Explore alternative scaling of process and observation noise for robustness  
- **Extend to multi-asset pairs or ETFs for portfolio-level strategies  
- **Conduct comprehensive backtesting with transaction costs  
- **Incorporate dynamic volatility metrics (e.g., VIX) for adaptive process noise  

---

## Tools & Libraries
- **Python 3.x  
- **[polars](https://www.pola.rs/) for data manipulation  
- **[yfinance](https://pypi.org/project/yfinance/) for stock data  
- **[statsmodels](https://www.statsmodels.org/) for statistical tests and regressions  
- **[matplotlib](https://matplotlib.org/) for plotting  
- **[numpy](https://numpy.org/) & [pandas](https://pandas.pydata.org/) for array and DataFrame operations