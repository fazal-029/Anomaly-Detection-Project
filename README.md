# Abstract

This paper addresses the detection of anomalies in time-series data by identifying records that deviate from
expected patterns such as trends and seasonality. We present a comprehensive survey of anomaly detection
techniques, introducing a classification framework that categorizes methods based on anomaly types and
analysis techniques. We classify the techniques into anomalous record detection and anomalous sequence
detection methods, and the computational methods are categorized into statistical, machine learning, deep
learning, and time series decomposition techniques. This classification was performed based on an extensive
review of existing literature in anomaly detection. Existing surveys rely on datasets that are flawed due
to unrealistic assumptions, limited coverage of anomaly types, or inadequate representation of real-world
scenarios, leading to unreliable results. Additionally, the common use of precision, recall, and F1 score for
evaluation is inappropriate for time series analysis because these metrics do not account for the temporal
dependencies in time series data, often leading to misleading assessments. To address these issues, we utilize
datasets that account for these typical flaws and employ evaluation metrics specifically designed for time
series, resulting in more reliable outcomes.
The paper conducts an empirical comparison of 13 anomaly detection techniques using univariate UCR
Anomaly Archive and multivariate Exathlon datasets, which are well-suited for capturing time series char-
acteristics in real-world scenarios. We have used a customized accuracy score for the UCR dataset and
Range-AUC-PR for the Exathlon dataset. These metrics are tailored for time series datasets and ensure a
more reliable assessment. Our analysis identifies the LSTM autoencoder as the most effective for detecting
anomalous records in univariate data, and the Transformer model as the most effective for multivariate data,
due to their abilities to capture long-term dependencies and nonlinear associations. This study offers valuable
insights into the strengths and limitations of various techniques, serving as a useful reference for researchers
and practitioners in anomaly detection
