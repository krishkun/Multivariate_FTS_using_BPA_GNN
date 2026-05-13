% ================================================
% --- DATASETS SECTION ---
% ================================================
\chapter{Experimental Setup}
\section{Datasets}
\label{sec:datasets}

To evaluate the performance of the proposed model, three widely used real-world multivariate time series datasets with diverse characteristics are utilized: Electricity Load Diagrams, PeMS Traffic, and MPI Jena Weather.

\subsection{Electricity Load Diagrams (2011-2014)}
\label{subsec:electricity}
The electricity consumption data for 370 clients in Portugal from 2011 to 2014 is contained in this dataset \cite{uci_electricity}. Measurements were recorded every 15 minutes. 

[Image of electricity load consumption daily and seasonal patterns graph]
 A different time series variable is represented by each client, by which a high-dimensional multivariate dataset suitable for analyzing electricity demand patterns and forecasting consumption is constituted. Temporal dependencies influenced by daily, weekly, and seasonal cycles, as well as potential external factors, are captured by the dataset.

\subsection{PeMS Traffic Data}
\label{subsec:traffic}
Extensive traffic information collected from sensors deployed across California highways is provided by the Caltrans Performance Measurement System (PeMS) dataset \cite{pems_traffic}. A subset of this data is utilized, wherein traffic flow (vehicles per time unit), occupancy, or speed measurements recorded at regular intervals (e.g., every 5 minutes) from a network of sensor stations are typically focused upon.  Each sensor station is treated as a variable in the multivariate time series. This dataset is characterized by complex spatio-temporal correlations, by which traffic congestion patterns influenced by time of day, day of week, incidents, and geographical proximity between sensors are reflected.

\subsection{MPI Jena Weather Data}
\label{subsec:weather}
This dataset includes meteorological measurements gathered at the Max Planck Institute for Biogeochemistry in Jena, Germany \cite{jena_weather}.  The dataset comprises multiple weather parameters, including temperature, atmospheric pressure, humidity, wind direction, and wind speed, recorded at high frequency (approximately every 10 minutes) over several years (e.g., 2009-2016).   Each weather parameter constitutes a variable in the multivariate time series.  This dataset is valuable for examining the complex dynamics and interdependencies among various meteorological variables and for applications such as weather forecasting.

\section{Experiment}
\label{sec:experimental_setup}

The experimental configuration to evaluate the proposed Fuzzy BPA-EGNN model for multivariate time series classification is as detailed below in the figure:

% ==========================================
% --- START: ADDED METHODOLOGY FIGURE ---
% ==========================================
\begin{figure}[htbp]
    \centering
    \resizebox{\textwidth}{!}{%
    \begin{tikzpicture}[
        node distance=1.5cm,
        auto,
        block/.style={
            rectangle, 
            draw, 
            fill=blue!5, 
            text width=3cm, 
            text centered, 
            rounded corners, 
            minimum height=1.5cm,
            font=\small
        },
        line/.style={
            draw, 
            -latex, 
            thick
        },
        container/.style={
            draw, 
            dashed, 
            inner sep=0.5cm, 
            rounded corners, 
            fill=gray!5,
            label={[anchor=north, font=\bfseries]north:#1}
        }
    ]

    % Nodes
    \node [block, fill=green!10] (input) {Multivariate Time Series Input};
    \node [block, right=of input] (preprocess) {Preprocessing\\(Normalization \& Windowing)};
    \node [block, right=of preprocess] (graph) {Graph Construction\\(Node \& Edge Definition)};
    
    \node [block, below=of graph, fill=orange!10] (fuzzy) {Fuzzy BPA Module\\(Fuzzification \& Belief Mass Assignment)};
    \node [block, left=of fuzzy, fill=orange!10] (egnn) {Edge-GNN Layer\\(Message Passing \& Aggregation)};
    \node [block, left=of egnn, fill=red!10] (output) {Classification / Forecasting Output};

    % Edges
    \path [line] (input) -- (preprocess);
    \path [line] (preprocess) -- (graph);
    \path [line] (graph) -- (fuzzy);
    \path [line] (fuzzy) -- (egnn);
    \path [line] (egnn) -- (output);

    % Container Box for Core Model
    \begin{scope}[on background layer]
        \node [container={Proposed Fuzzy BPA-EGNN Architecture}, fit=(graph) (fuzzy) (egnn)] (model_box) {};
    \end{scope}

    \end{tikzpicture}
    }
    \caption{Flowchart of the proposed Fuzzy BPA-EGNN methodology. The process transitions from raw multivariate time series input, through graph construction and fuzzy belief assignments, to the final edge-labeling GNN classification.}
    \label{fig:methodology_flow}
\end{figure}
% ==========================================
% --- END: ADDED METHODOLOGY FIGURE ---
% ==========================================

\subsection{Datasets and Preprocessing}
The proposed model is assessed using the three multivariate time series datasets outlined in Section~\ref{sec:datasets}: Electricity Load Diagrams \cite{uci_electricity}, PeMS Traffic \cite{pems_traffic}, and MPI Jena Weather \cite{jena_weather}. A representative subset and corresponding classification task are established for each dataset, such as classifying traffic congestion levels, predicting specific weather events, or identifying anomalous electricity usage patterns. Before model input, the time series data for each variable are normalised to a defined range (e.g., [0, 1] or [-1, 1] through min-max scaling, or standardised via z-score normalisation) to ensure stable training.  The datasets are divided into standard training, validation, and testing sets, usually employing a chronological split (e.g., 70% training, 10% validation, 20% testing).

\subsection{Implementation Details}
The proposed model is implemented using Python with standard scientific libraries such as NumPy and PyTorch, potentially leveraging GNN-specific libraries like PyTorch Geometric. All experiments are conducted on hardware equipped with NVIDIA GPUs (e.g., Tesla V100 or similar) so that training is accelerated.

\subsubsection{Graph Construction}
Input time series are segmented into windows of a predefined length $L$. For each window, features are extracted (e.g., mean, standard deviation, FFT coefficients, or raw segment values flattened/pooled) and assigned to nodes in a graph. Edges are initially constructed based on a predefined neighborhood (e.g., fully connected within a window or based on channel correlation/proximity) or learned dynamically.

\subsubsection{Fuzzy BPA Module}
Triangular or Gaussian fuzzy membership functions are utilized by the fuzzy BPA module to represent linguistic concepts (e.g., "low," "medium," "high" value/change) for node features. 

[Image of triangular fuzzy membership function]
 The belief mass assignments are calculated based on these memberships and predefined or learned rules derived from evidence theory principles. Parameters controlling the fuzzification and belief combination (e.g., number of fuzzy sets, combination rules like Dempster's rule) are tuned using the validation set.


\subsection{Evaluation Metrics}
To evaluate the forecasting performance of the proposed model, two widely used metrics are utilized: Mean Squared Error (MSE) and Mean Absolute Error (MAE). The deviation between the predicted values $\hat{Y}$ and the ground truth values $Y$ is quantified by these metrics. They are defined as follows:

\begin{itemize}
    \item Mean Squared Error (MSE): The average squared difference between the estimated values and the actual value is measured. Larger errors are penalized more significantly. 
    \begin{equation}
        MSE = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2
    \end{equation}
    
    \item Mean Absolute Error (MAE): The average magnitude of the errors in a set of predictions is measured, without considering their direction.
    \begin{equation}
        MAE = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|
    \end{equation}
\end{itemize}

Where $N$ represents the number of samples, $y_i$ is the actual value, and $\hat{y}_i$ is the predicted value. Better predictive performance is indicated by lower values for both MSE and MAE. The model is evaluated across three different prediction horizons: $L_{pred} \in \{96, 192, 336\}$.

\subsection{Baseline Methods}
Comparisons are made against an extensive collection of cutting-edge time series forecasting models, which are divided into the following categories, in order to confirm the efficacy of the suggested framework:

 \begin{itemize}  
 \item Transformer-based Techniques:  Among the models are iTransformer~\cite{iTransformer}, which embeds the entire time series by inverting the transformer structure;  PatchTST~\cite{PatchTST}, which makes use of channel independence and patching;  Crossformer~\cite{Crossformer}, which captures cross-dimension dependency;  ETSformer~\cite{ETSformer}, which makes use of exponential smoothing;  Autoformer~\cite{Autoformer}, which has auto-correlation mechanisms; FEDformer~\cite{FEDformer}, which uses frequency enhanced decomposition; and the Non-stationary Transformer (Stationary)~\cite{Stationary}.
    
\item MLP-based and linear approaches:  TiDE~\cite{TiDE}, a dense encoder-decoder model; LightTS~\cite{LightTS}, a lightweight sampling-oriented MLP; and DLinear~\cite{DLinear} and Rlinear~\cite{Rlinear}, two straightforward but powerful linear models, are compared.
    
\item CNN and Other Architectures: This category includes TimesNet~\cite{TimesNet}, by which temporal 2D-variations are modeled, and TEFN~\cite{zhan2024tefn}, a recent text-enhanced forecasting network.
\end{itemize}

These baselines represent the current state-of-the-art in multivariate time series forecasting, covering various architectural paradigms including Transformers, MLPs, and CNNs.

% ================================================
% --- ALTERNATIVE EVIDENCE MODULES ---
% ================================================
\section{Alternative Evidence Theory Modules}
\label{sec:alt_evidence_modules}

This section presents alternative evidence theory modules implemented and compared in this work, extending the basic Fuzzy BPA-EGNN framework.

\subsection{Transferable Belief Model (TBM)}
\label{subsec:tbm_module}

The Transferable Belief Model (TBM) is a two-level model proposed by Smets:

\subsubsection{Credal Level}

At the credal level, beliefs are represented and combined without normalization:
\begin{equation}
    m_1 \cap m_2(A) = \sum_{B \cap C = A} m_1(B) \cdot m_2(C)
\end{equation}

The open-world assumption allows for mass on the empty set, representing the possibility of unknown hypotheses.

\subsubsection{Pignistic Level}

At the pignistic level, beliefs are transformed to probabilities for decision making:
\begin{equation}
    \text{BetP}(\omega) = \sum_{A \ni \omega} \frac{m(A)}{|A|}
\end{equation}

\subsubsection{Conflict Management}

TBM handles conflict by:
\begin{itemize}
    \item Keeping conflict as mass on empty set (open world)
    \item Using conjunctive combination without normalization
    \item Providing pignistic transformation for decisions
\end{itemize}

\subsection{Pignistic Transformation Module}
\label{subsec:pignistic_module}

The Pignistic Transformation Module provides tools for decision making under uncertainty:

\subsubsection{Forward Transformation}

Transform BPA to probability:
\begin{equation}
    \text{BetP}(\omega_i) = m(\{\omega_i\}) + \frac{m(\emptyset)}{n}
\end{equation}

\subsubsection{Pignistic Distance}

Distance between probability distributions:
\begin{equation}
    d(\text{BetP}_1, \text{BetP}_2) = \|\text{BetP}_1 - \text{BetP}_2\|_2
\end{equation}

\subsubsection{Decision Criteria}

\begin{itemize}
    \item Maximum Belief: Choose hypothesis with highest belief
    \item Maximum Plausibility: Choose hypothesis with highest plausibility
    \item Maximum Pignistic Probability: Choose hypothesis with highest BetP
\end{itemize}

\subsection{Evidential k-NN Module}
\label{subsec:eknn_module}

The Evidential k-NN Module combines k-nearest neighbors with evidence theory:

\subsubsection{Distance to Mass Conversion}

Each neighbor provides evidence based on distance:
\begin{equation}
    m_i(\omega_j) = \alpha \cdot \exp(-\gamma \cdot d_i^2) \quad \text{if } y_i = \omega_j
\end{equation}
\begin{equation}
    m_i(\Omega) = 1 - m_i(\omega_j)
\end{equation}

where $\alpha \in [0,1]$ controls the certainty and $\gamma > 0$ controls the distance sensitivity.

\subsubsection{Evidence Combination}

Masses from all neighbors are combined using Dempster's rule:
\begin{equation}
    m = m_1 \oplus m_2 \oplus \cdots \oplus m_k
\end{equation}

\subsection{Credal Classification Module}
\label{subsec:credal_module}

Credal Classification provides imprecise classification for robust decisions:

\subsubsection{Credal Set}

A credal set is a set of probability distributions characterized by belief and plausibility:
\begin{equation}
    \mathcal{P} = \{P : \text{Bel}(A) \leq P(A) \leq \text{Pl}(A), \forall A\}
\end{equation}

\subsubsection{Decision Rules}

\begin{enumerate}
    \item \textbf{Precise Classification}: If $\text{Bel}(\omega_i) > \theta$, assign to class $\omega_i$
    
    \item \textbf{Imprecise Classification}: If multiple classes have similar belief, assign to set of classes
    
    \item \textbf{Rejection}: If uncertainty is too high, reject classification
\end{enumerate}

\subsubsection{Credal Distance}

Hausdorff distance between credal sets:
\begin{equation}
    d_H(\mathcal{P}_1, \mathcal{P}_2) = \max\left\{\sup_{P_1 \in \mathcal{P}_1} \inf_{P_2 \in \mathcal{P}_2} d(P_1, P_2), \sup_{P_2 \in \mathcal{P}_2} \inf_{P_1 \in \mathcal{P}_1} d(P_1, P_2)\right\}
\end{equation}

% ================================================
% --- EXPLAINABILITY TOOLS ---
% ================================================
\section{Explainability Tools}
\label{sec:explainability_tools}

\subsection{Belief Visualization}
\label{subsec:belief_viz}

Visualization tools for evidence and belief functions:
\begin{itemize}
    \item BPA bar charts
    \item Belief-Plausibility interval plots
    \item Evidence combination visualization
    \item Uncertainty distribution histograms
\end{itemize}

\subsection{Uncertainty Metrics}
\label{subsec:uncertainty_metrics_impl}

\textbf{Total Uncertainty:}
\begin{equation}
    TU = \text{Pl}(A) - \text{Bel}(A)
\end{equation}

\textbf{Non-specificity (Yager's measure):}
\begin{equation}
    N(m) = \sum_{A \subseteq \Omega} m(A) \log_2 |A|
\end{equation}

\textbf{Conflict Measure:}
\begin{equation}
    C(m) = -\sum_{\omega \in \Omega} m(\omega) \log m(\omega)
\end{equation}

\textbf{Ambiguity Measure:}
\begin{equation}
    A(m) = 1 - \max_{\omega} \text{Bel}(\omega)
\end{equation}

\subsection{SHAP-based Feature Importance}
\label{subsec:shap_feature}

SHAP (SHapley Additive exPlanations) values provide feature importance:
\begin{equation}
    \phi_i = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]
\end{equation}

\subsection{Evidence Decomposition}
\label{subsec:evidence_decomp}

Decompose predictions into evidence contributions:
\begin{itemize}
    \item Temporal contribution analysis
    \item Variable contribution analysis
    \item Evidence source attribution
\end{itemize}

% ================================================
% --- COMPARISON FRAMEWORK ---
% ================================================
\section{Comparison Framework for Evidence Methods}
\label{sec:comparison_framework}

\subsection{Evidence Combination Methods Compared}
\label{subsec:evidence_methods_compared}

We compare the following evidence combination methods:
\begin{enumerate}
    \item Dempster's Rule (baseline)
    \item Murphy's Average Rule
    \item Yager's Rule
    \item TBM Conjunctive Rule
\end{enumerate}

\subsection{Evaluation Criteria}
\label{subsec:eval_criteria}

\begin{itemize}
    \item Prediction accuracy (MSE, MAE)
    \item Uncertainty calibration
    \item Decision quality (precise vs. imprecise)
    \item Computational efficiency
\end{itemize}
