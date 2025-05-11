""" 
Set of routines for the analysis of multiple streams of data produced with the nested sampling algorithm.
"""

import numpy as np

class StreamsAnalyser:
    def __init__(self, beta_ref, beta_range):
        self.beta_ref = beta_ref
        self.beta_range = beta_range
    ####
    def run(self, parallelize, ssa_list, wi_strategy="symm"):
        """
        n_streams analysis using a list of ssa (single-stream-analiser).
        The length of ss_list equals n_streams

        """
        streams_logX, streams_S, streams_logS = [], [], []
        streams_wL = []
        streams_Z_curve = []
        streams_log_rho = []
        streams_P_avg = []
        streams_P2_avg = []
        i_stream = 0
        n_streams = len(ssa_list)
        dt_wL = 0
        dt_logZ = 0
        for ssa in ssa_list:
            print("Analysing streams: ", 100*i_stream/n_streams, "%")
            S = ssa.S
            streams_S.append(S)
            logS = np.log(S)
            streams_logS.append(logS)
            ssa_beta_ref = ssa.run_analysis_beta_ref(parallelize=parallelize, beta_ref=self.beta_ref, wi_strategy=wi_strategy)

            logX = ssa_beta_ref["logX"]
            streams_logX.append(logX)
            # log_w = ssa_beta_ref["log_w"]
            # logL = ssa_beta_ref["logL"]

            log_wL = ssa_beta_ref["log_wL"]
            wL = np.exp(log_wL)
            streams_wL.append(wL)

            # idx_prn = ssa_beta_ref["idx_pruned_interval"]
            log_Z_curve = ssa_beta_ref["log_Z_curve"]
            Z_curve = np.exp(log_Z_curve)
            streams_Z_curve.append(Z_curve)

            log_rho = ssa.get_log_rho() ## density of states
            streams_log_rho.append(log_rho)

            ## Plaquette expectation value
            P_avg = ssa.get_average_plaquette(beta_range=self.beta_range)
            streams_P_avg.append(P_avg)

            P2_avg = ssa.get_average_P2(beta_range=self.beta_range)
            streams_P2_avg.append(P2_avg)
            i_stream += 1
            #
            dt_wL += ssa_beta_ref["dt_wL"]
            dt_logZ += ssa_beta_ref["dt_logZ"]
        ####
        streams_logX    = np.array(streams_logX   )
        streams_S       = np.array(streams_S      )
        streams_logS    = np.array(streams_logS   )
        streams_wL      = np.array(streams_wL     )
        streams_Z_curve = np.array(streams_Z_curve)
        streams_log_rho = np.array(streams_log_rho)
        streams_P_avg   = np.array(streams_P_avg  )
        streams_P2_avg   = np.array(streams_P2_avg  )
        res = dict({
            "logX": streams_logX,
            "logS": streams_logS,
            "wL": streams_wL,
            "Z_curve": streams_Z_curve,
            "log_rho": streams_log_rho,
            "P_avg": streams_P_avg,
            "P2_avg": streams_P2_avg,
            "dt_wL" : dt_wL,
            "dt_logZ" : dt_logZ
        })
        return res
    ####
####
        

