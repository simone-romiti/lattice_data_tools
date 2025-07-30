""" Charge factors in front to be added in front of the Vector-Vector correlator """

from lattice_data_tools.constants import q_dict, e2

q2_u = q_dict["u"]**2
q2_d = q_dict["d"]**2
q2_s = q_dict["s"]**2
q2_c = q_dict["c"]**2

Q2 = {"light": e2*(q2_u + q2_d), "s": e2*(q2_s), "c": e2*(q2_c)}
