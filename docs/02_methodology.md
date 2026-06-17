# ROGII Wellbore Geology Prediction: Methodology

## 1. Project Objective & Context

The core objective of this project is to accurately predict the True Vertical Thickness (`TVT`) of geological formations intersected by horizontal wellbores. In directional drilling operations, horizontal wells penetrate reservoir layers "blind"—relying on delayed downhole telemetry and measurement-while-drilling (MWD) tools.

Bridging the structural gap between known vertical control points (Type Wells) and active horizontal trajectories requires modeling non-linear physical paths, stratigraphic boundaries, and complex log-trajectory interactions. The target variable `TVT` represents a critical geometric constraint for reservoir geosteering, structural modeling, and resource estimation.

## 2. Exploratory Data Analysis (EDA) Findings

- **Data Volume & Cardinality**: 5,092,255 records across 14 initial parameters, presenting high sequence continuity.
- **Data Quality Matrix**: 0% missing data fields across critical telemetry streams; no extensive synthetic imputation arrays or iterative predictive backfills are required.
- **Target Distribution Profile (`TVT`)**: Real-valued continuous distribution with a mean metric of 11,503 ft and a standard deviation ($\sigma$) of $\approx 640\text{ ft}$.
- **Geological Markers Identified**: Signal anomalies indicate intersections with prominent regional formation horizons, including the Buda Limestone (`BUDA`), Austin Chalk sub-members (`ASTNU`, `ASTNL`), and the Upper/Lower Eagle Ford Shale (`EGFDU`, `EGFDL`).
- **Grouping Dynamic**: The data matrix encapsulates 773 uniquely identified operational wellbores (`well_id`), forming distinct spatial-structural clusters.

## 3. Validation Strategy: Leakage Prevention

Geostatistical spatial data displays a profound degree of spatial autocorrelation; measurements taken in close proximity along a single well path inherit nearly identical geological characteristics. Utilizing a standard randomized or stratified `K-Fold` splitting routine causes extreme, catastrophic data leakage—the training set effectively "memorizes" adjacent sequence data points along the exact same wellbore.

- **Protocol**: 5-Fold `GroupKFold`.
- **Grouping Anchor**: `well_id`. By isolating entire well trajectories within individual validation folds, this protocol mirrors true operational deployment: the model is continuously evaluated on completely unseen, blind well assets, preventing optimistic local metric bias.

## 4. Geostatistical Feature Engineering

To capture subsurface structural mechanics and guide the gradient boosters beyond raw coordinate lines, the feature space is geometrically expanded using Polars over sequentially sorted windows.

### 4.1 Type Well Classification

Vertical control wells contain the definitive stratigraphy profile, whereas horizontal tracks represent exploratory paths.
$$\text{Is Type Well} = \begin{cases} 1 & \text{if } \frac{\max(Z)}{\max(MD)} > 0.95 \\ 0 & \text{otherwise} \end{cases}$$

### 4.2 Spatial & Geometric Gradients

Instead of tracking absolute global coordinates, the model processes the multi-dimensional derivatives of the well path:

- **Directional Velocity Components**: $\frac{dX}{dMD}, \frac{dY}{dMD}, \frac{dZ}{dMD}$ to track structural dip changes and bit inclination.
- **Horizontal 2D Displacement**: $\text{Disp}_{\text{horiz}} = \sqrt{dX^2 + dY^2}$.
- **Tortuosity Proxy (Dogleg Severity Approximation)**: $\frac{\sqrt{dX^2 + dY^2 + dZ^2}}{dMD}$. Ratios deviating significantly from 1 indicate tight structural trajectory steering, a proxy for geosteering corrections along strict formation boundaries.

### 4.3 Stratigraphic Context (Lags & Leads)

Rock formations display sequence dependencies. To mimic classic sequence stratigraphy mapping, localized rolling structures are engineered over the Gamma Ray (`GR`) log:

- **Rolling Sequence Shifts**: Lags and leads calculated at precise offsets of $[-5, -2, -1, 1, 2, 5]$ rows capture boundaries and trends in rock density.

### 4.4 Log-Trajectory Interactions

Fusing downhole physical sensors directly with the spatial mechanics of the drill string:

- **Lithology Gradient**: $\frac{dGR}{dMD}$ (rate of change in shale content/gamma intensity).
- **Trajectory Interaction Feature**: $\text{GR\_x\_gradZ} = \text{GR} \times \frac{dZ}{dMD}$. This feature acts as a crucial geometric decoupling mechanism: it helps the model differentiate whether the sensor encountered a real, steep geological formation dip or if the drill bit simply initiated a sharp ascent/descend through a flat stratum.

## 5. Modeling Architecture & Optimization

- **Core Framework**: `WellboreModelLGBM` wrapping native LightGBM Gradient Boosted Decision Tree (GBDT) structures.
- **Hardware Interactivity**: Native OpenCL/CUDA tree-building compilation flags target the local RTX 5080 Laptop GPU, minimizing split-finding evaluation time.
- **Hyperparameter Optimization**: Automated execution via `Optuna` using a Tree-structured Parzen Estimator (TPE) sampler over 50 iterations. The tuning space prioritizes aggressive regularization parameters to compress leaf nodes and explicitly penalize complex splits that overfit localized wellbore spaces.
