"""Reference database, Elsevier numbered style.  Keys are cited as {cite:key}.

Entries carried over from the original draft were converted from IEEE style;
entries added for the journal version were checked against Crossref / the
publisher record (see paper/REVISION_NOTES.md).
"""

REFS = {
    # ---------------------------------------------------------- original draft
    "mishra2024": "N. Mishra, S.H. Islam, S. Zeadally, A survey on security and cryptographic perspective of Industrial-Internet-of-Things, Internet Things 25 (2024) 101037. https://doi.org/10.1016/j.iot.2023.101037",
    "xing2024": "W. Xing, J. Shen, Security control of cyber–physical systems under cyber attacks: a survey, Sensors 24 (2024) 3815. https://doi.org/10.3390/s24123815",
    "reis2025": "M.J.C.S. Reis, Lightweight signal processing and edge AI for real-time anomaly detection in IoT sensor networks, Sensors 25 (2025) 6629. https://doi.org/10.3390/s25216629",
    "ahmad2024": "R. Ahmad, Enhanced drift self-calibration of low-cost sensor networks based on cluster and advanced statistical tools, Measurement 236 (2024) 115158. https://doi.org/10.1016/j.measurement.2024.115158",
    "schaller2025": "M. Schaller, M. Kruse, A. Ortega, M. Lindauer, B. Rosenhahn, AutoML for multi-class anomaly compensation of sensor drift, Measurement 250 (2025) 117097. https://doi.org/10.1016/j.measurement.2025.117097",
    "shen2024": "Y. Shen, Z. Qin, Detection, differentiation and localization of replay attack and false data injection attack based on random matrix, Sci. Rep. 14 (2024) 2758. https://doi.org/10.1038/s41598-024-52954-z",
    "yang2024": "H. Yang, D. Gu, J. He, A robust and efficient federated learning algorithm against adaptive model poisoning attacks, IEEE Internet Things J. 11 (2024) 16289–16302. https://doi.org/10.1109/JIOT.2024.3351371",
    "sahu2025": "C. Sahu, S. Poddar, Acoustic attack mitigation approach for MEMS inertial sensors using change point detection on MhIMU framework, IEEE Trans. Aerosp. Electron. Syst. 61 (2025) 743–752. https://doi.org/10.1109/TAES.2024.3447626",
    "xu2024": "L. Xu, Z. Han, D. Zhao, X. Li, F. Yu, C. Chen, Addressing concept drift in IoT anomaly detection: drift detection, interpretation, and adaptation, IEEE Trans. Sustain. Comput. 9 (2024) 913–924. https://doi.org/10.1109/TSUSC.2024.3386667",
    "darban2025": "Z. Zamanzadeh Darban, G.I. Webb, S. Pan, C. Aggarwal, M. Salehi, Deep learning for time series anomaly detection: a survey, ACM Comput. Surv. 57 (2025) 1–42. https://doi.org/10.1145/3691338",
    "lee2025": "J. Lee, J. Lee, S.B. Kim, Uncertainty-informed dynamic threshold for time series anomaly detection, Expert Syst. Appl. 278 (2025) 127379. https://doi.org/10.1016/j.eswa.2025.127379",
    "yap2024": "Y.S. Yap, M.R. Ahmad, Modified overcomplete autoencoder for anomaly detection based on TinyML, IEEE Sens. Lett. 8 (2024) 1–4. https://doi.org/10.1109/LSENS.2024.3463977",
    "ino2024": "T. Ino, K. Yoshida, H. Matsutani, T. Fujino, Data poisoning attack against neural network-based on-device learning anomaly detector by physical attacks on sensors, Sensors 24 (2024) 6416. https://doi.org/10.3390/s24196416",
    "saber2025": "A.M. Saber, A. Maheshwari, A. Youssef, D. Kundur, Adversarial attacks on deep learning-based false data injection detection in differential relays, IEEE Trans. Smart Grid 16 (2025) 4155–4166. https://doi.org/10.1109/TSG.2025.3579316",
    "ullah2025": "S. Ullah, M.F. Siddique, J.-M. Kim, Multi-sensor observer-based residual learning with auto-permutation feature importance for fault diagnosis of multistage centrifugal pumps under variable pressures, Sci. Rep. 15 (2025) 45735. https://doi.org/10.1038/s41598-025-32726-z",
    "wu2024": "Y. Wu, B. Sicard, S.A. Gadsden, Physics-informed machine learning: a comprehensive review on applications in anomaly detection and condition monitoring, Expert Syst. Appl. 255 (2024) 124678. https://doi.org/10.1016/j.eswa.2024.124678",
    "zhu2025": "Q. Zhu, Y. Ding, J. Jiang, S.-H. Yang, Anomaly detection using invariant rules in industrial control systems, Control Eng. Pract. 154 (2025) 106164. https://doi.org/10.1016/j.conengprac.2024.106164",
    "liu2025": "C. Liu, Y. Shi, S. Zhou, L. Xu, Y. Li, Distinguishable attack and fault detection in interconnected cyber–physical systems, Control Eng. Pract. 156 (2025) 106216. https://doi.org/10.1016/j.conengprac.2024.106216",
    "hu2024": "Y. Hu, X. Dai, D. Cui, Q. Liu, Anomaly identification for cyber-physical systems subject to replay attacks and sensor faults, IEEE Trans. Circuits Syst. II Express Briefs 71 (2024) 5044–5048. https://doi.org/10.1109/TCSII.2024.3349777",
    "radovanovic2026": "D. Radovanovic, N. Savanovic, P. Kresoja, J. Janackovic, T. Petrovic, PhySec-Edge: a hybrid physics-informed and edge AI framework for anomaly detection in industrial IoT sensor networks, J. Sens. Actuator Netw. 15 (2026) 58. https://doi.org/10.3390/jsan15040058",
    "zideh2024": "M.J. Zideh, P. Chatterjee, A.K. Srivastava, Physics-informed machine learning for data anomaly detection, classification, localization, and mitigation: a review, challenges, and path forward, IEEE Access 12 (2024) 4597–4617. https://doi.org/10.1109/ACCESS.2023.3347989",
    "sayghe2025": "A. Sayghe, Digital twin-driven intrusion detection for industrial SCADA: a cyber-physical case study, Sensors 25 (2025) 4963. https://doi.org/10.3390/s25164963",
    "kallesoe2024": "C.S. Kallesøe, R. Wisniewski, Cyber-attack and fault detection using a digital twin of the controller software, IFAC-PapersOnLine 58 (2024) 97–102. https://doi.org/10.1016/j.ifacol.2024.07.200",
    # ------------------------------------------------- added for the journal version
    "korycki2023": "Ł. Korycki, B. Krawczyk, Adversarial concept drift detection under poisoning attacks for robust data stream mining, Mach. Learn. 112 (2023) 4013–4048. https://doi.org/10.1007/s10994-022-06177-w",
    "kravchik2021": "M. Kravchik, B. Biggio, A. Shabtai, Poisoning attacks on cyber attack detectors for industrial control systems, in: Proc. 36th Annu. ACM Symp. Appl. Comput. (SAC '21), ACM, 2021, pp. 116–125. https://doi.org/10.1145/3412841.3441892",
    "kravchik2022": "M. Kravchik, L. Demetrio, B. Biggio, A. Shabtai, Practical evaluation of poisoning attacks on online anomaly detectors in industrial control systems, Comput. Secur. 122 (2022) 102901. https://doi.org/10.1016/j.cose.2022.102901",
    "shukla2023": "R.M. Shukla, S. Sengupta, A novel machine learning pipeline to detect malicious anomalies for the Internet of Things, Internet Things 20 (2022) 100603. https://doi.org/10.1016/j.iot.2022.100603",
    "hinder2025": "F. Hinder, V. Vaquet, B. Hammer, Adversarial attacks for drift detection, in: Proc. Eur. Symp. Artif. Neural Netw. Comput. Intell. Mach. Learn. (ESANN 2025), Bruges, 2025, pp. 555–560. https://doi.org/10.48550/arXiv.2411.16591",
    "ghafouri2018": "A. Ghafouri, Y. Vorobeychik, X. Koutsoukos, Adversarial regression for detecting attacks in cyber-physical systems, in: Proc. 27th Int. Joint Conf. Artif. Intell. (IJCAI-18), 2018, pp. 3769–3775. https://doi.org/10.24963/ijcai.2018/524",
    "morand2024": "N. Morand, A. Boubrima, W. Bechkit, Z. Shaikhanov, HeatPulse: thermal attacks on air pollution sensors, in: SPICES Workshop, co-located with EWSN 2024, Abu Dhabi, 2024. https://ewsn.org/file-repository/ewsn2024/SPICES_2024_CamReady_9.pdf",
    "intel2004": "P. Bodik, W. Hong, C. Guestrin, S. Madden, M. Paskin, R. Thibaux, Intel Lab Data, Intel Berkeley Research Lab, 2004. http://db.csail.mit.edu/labdata/labdata.html (accessed 12 October 2026)",
    "alduchov1996": "O.A. Alduchov, R.E. Eskridge, Improved Magnus form approximation of saturation vapor pressure, J. Appl. Meteorol. 35 (1996) 601–609. https://doi.org/10.1175/1520-0450(1996)035<0601:IMFAOS>2.0.CO;2",
    "liu2008iforest": "F.T. Liu, K.M. Ting, Z.-H. Zhou, Isolation forest, in: Proc. 8th IEEE Int. Conf. Data Min. (ICDM 2008), IEEE, 2008, pp. 413–422. https://doi.org/10.1109/ICDM.2008.17",
    "page1954": "E.S. Page, Continuous inspection schemes, Biometrika 41 (1954) 100–115. https://doi.org/10.1093/biomet/41.1-2.100",
    "breiman2001": "L. Breiman, Random forests, Mach. Learn. 45 (2001) 5–32. https://doi.org/10.1023/A:1010933404324",
    "holm1979": "S. Holm, A simple sequentially rejective multiple test procedure, Scand. J. Stat. 6 (1979) 65–70. https://www.jstor.org/stable/4615733",
    "wilcoxon1945": "F. Wilcoxon, Individual comparisons by ranking methods, Biom. Bull. 1 (1945) 80–83. https://doi.org/10.2307/3001968",
    "kalman1960": "R.E. Kalman, A new approach to linear filtering and prediction problems, J. Basic Eng. 82 (1960) 35–45. https://doi.org/10.1115/1.3662552",
    "gama2014": "J. Gama, I. Žliobaitė, A. Bifet, M. Pechenizkiy, A. Bouchachia, A survey on concept drift adaptation, ACM Comput. Surv. 46 (2014) 1–37. https://doi.org/10.1145/2523813",
    "liu2009fdi": "Y. Liu, P. Ning, M.K. Reiter, False data injection attacks against state estimation in electric power grids, in: Proc. 16th ACM Conf. Comput. Commun. Secur. (CCS '09), ACM, 2009, pp. 21–32. https://doi.org/10.1145/1653662.1653666",
    "mo2009": "Y. Mo, B. Sinopoli, Secure control against replay attacks, in: Proc. 47th Annu. Allerton Conf. Commun. Control Comput., IEEE, 2009, pp. 911–918. https://doi.org/10.1109/ALLERTON.2009.5394956",
    "isermann2005": "R. Isermann, Model-based fault-detection and diagnosis – status and applications, Annu. Rev. Control 29 (2005) 71–85. https://doi.org/10.1016/j.arcontrol.2004.12.002",
    "liu2026calib": "L. Liu, Q. Zhang, K.G. Shin, Adversarial calibration attack on autonomous vehicles, arXiv preprint arXiv:2608.28778, 2026. https://doi.org/10.48550/arXiv.2608.28778",
}
