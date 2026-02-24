import pennylane as qml
import torch

from app.settings import AppSettings
from app.vqe.base_module import BaseVQEModule


class UCCSDVQEModule(BaseVQEModule):
    def __init__(self, settings: AppSettings):
        super().__init__(settings)

        vqe_cfg = settings.vqe

        singles, doubles = qml.qchem.excitations(self.n_electrons, self.n_qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

        self.s_wires = s_wires
        self.d_wires = d_wires

        n_params = len(s_wires) + len(d_wires)
        if n_params == 0:
            raise ValueError(
                "No UCCSD excitations generated (n_params=0). "
                "Check active_electrons/active_orbitals and molecule charge/multiplicity."
            )

        init = vqe_cfg.uccsd.init_theta_scale * torch.randn(
            n_params, dtype=torch.float64
        )
        self.theta = torch.nn.Parameter(init)

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(theta):
            qml.UCCSD(
                weights=theta,
                wires=range(self.n_qubits),
                s_wires=self.s_wires,
                d_wires=self.d_wires,
                init_state=self.hf_state,
            )
            return qml.expval(self.H)

        self._circuit = circuit
