"""The ELL1k model for approximately handling near-circular precessing orbits."""
from __future__ import absolute_import, division, print_function

import astropy.constants as c
import astropy.units as u
import numpy as np

from pint import GMsun, Tsun, ls

from .binary_generic import PSR_BINARY
from .ELL1_model import ELL1model


class ELL1kmodel(ELL1model):
    """This is a class for ELL1k pulsar binary model.

    ELL1k model is ELL1 model with 'exact' treatment of advance of periastron.
    ELL1k will behave differently than ELL1 when there is large advance of periastron.
    
    Ref : Abhimanyu Susobhanan et al., MNRAS, 480, 4, 5260–5271 (2018)
    """

    def __init__(self):
        super(ELL1kmodel, self).__init__()
        self.binary_name = "ELL1k"

        self.binary_delay_funcs = [self.ELL1kdelay]
        self.d_binarydelay_d_par_funcs = [self.d_ELL1kdelay_d_par]

    def tau(self):
        return self.ttasc() * self.OMDOT

    def eps1(self):
        return self.EPS1*np.cos(self.tau()) + self.EPS2*np.sin(self.tau())

    def d_eps1_d_EPS1(self):
        return np.cos(self.tau())
    
    def d_eps1_d_EPS2(self):
        return np.sin(self.tau())

    def d_eps1_d_TASC(self):
        return -self.OMDOT * self.eps2()

    def d_eps1_d_OMDOT(self):
        return self.ttasc() * self.eps2()

    def eps2(self):
        return self.EPS2*np.cos(self.tau()) - self.EPS1*np.sin(self.tau())

    def d_eps2_d_EPS2(self):
        return np.cos(self.tau())

    def d_eps2_d_EPS1(self):
        return -np.sin(self.tau())

    def d_eps2_d_TASC(self):
        return self.OMDOT * self.eps1()

    def d_eps2_d_OMDOT(self):
        return -self.ttasc() * self.eps1()

    def d_Dre_d_par(self, par):
        """Derivative computation.

        Computes::

            Dre = delayR = a1/c.c*(sin(phi) - 0.5* eps1*(cos(2*phi) + 3) +  0.5* eps2*sin(2*phi))
            d_Dre_d_par = d_a1_d_par /c.c*(sin(phi) - 0.5* eps1*(cos(2*phi) + 3) +  0.5* eps2*sin(2*phi)) +
                          d_Dre_d_Phi * d_Phi_d_par + d_Dre_d_eps1*d_eps1_d_par + d_Dre_d_eps2*d_eps2_d_par
        """
        a1 = self.a1()
        Phi = self.Phi()
        eps1 = self.eps1()
        eps2 = self.eps2()
        d_a1_d_par = self.prtl_der("a1", par)
        d_Dre_d_Phi = self.Drep()
        d_Phi_d_par = self.prtl_der("Phi", par)
        d_Dre_d_eps1 = a1 / c.c * (-0.5 * (np.cos(2 * Phi) + 3))
        d_Dre_d_eps2 = a1 / c.c * (0.5 * np.sin(2 * Phi))

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            d_Dre_d_par = (
                d_a1_d_par
                / c.c
                * (
                    np.sin(Phi)
                    - 0.5 * eps1 * (np.cos(2 * Phi) + 3)
                    + 0.5 * eps2 * np.sin(2 * Phi)
                )
                + d_Dre_d_Phi * d_Phi_d_par
                + d_Dre_d_eps1 * self.prtl_der("eps1", par)
                + d_Dre_d_eps2 * self.prtl_der("eps2", par)
            )
        return d_Dre_d_par
    
    
    
    def delayR(self):
        """ELL1k Roemer delay in proper time. Ch. Susobhanan et al., 2018 eq. 6 """
        Phi = self.Phi()
        return (
            self.a1()
            / c.c
            * (
                np.sin(Phi)
                + 0.5 * (self.eps2() * np.sin(2 * Phi) - self.eps1() * (np.cos(2 * Phi) + 3))
            )
        ).decompose()


    def ELL1kdelay(self):
        # TODO need add aberration delay
        return self.delayI() + self.delayS()

    def d_ELL1kdelay_d_par(self, par):
        return self.d_delayI_d_par(par) + self.d_delayS_d_par(par)
