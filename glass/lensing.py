# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
=====================================
Lensing fields (:mod:`glass.lensing`)
=====================================

.. currentmodule:: glass.lensing

Generators
==========

Single source plane
-------------------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   convergence
   shear
   ia_nla


Source distributions
--------------------

.. autosummary::
   :template: generator.rst
   :toctree: generated/

   lensing_dist


'''

import logging
import numpy as np
import healpy as hp

from .core import generator
from .util import restrict_interval


log = logging.getLogger(__name__)


@generator('kappa -> gamma1, gamma2')
def shear(lmax=None):
    r'''weak lensing shear from convergence

    Notes
    -----
    The shear field is computed from the convergence or deflection potential in
    the following way.

    Define the spin-raising and spin-lowering operators of the spin-weighted
    spherical harmonics as

    .. math::

        \eth {}_sY_{lm}
        = +\sqrt{(l-s)(l+s+1)} \, {}_{s+1}Y_{lm} \;, \\
        \bar{\eth} {}_sY_{lm}
        = -\sqrt{(l+s)(l-s+1)} \, {}_{s-1}Y_{lm} \;.

    The convergence field :math:`\kappa` is related to the deflection potential
    field :math:`\phi` as

    .. math::

        2 \kappa = \eth\bar{\eth} \, \phi = \bar{\eth}\eth \, \phi \;.

    The convergence modes :math:`\kappa_{lm}` are hence related to the
    deflection potential modes :math:`\phi_{lm}` as

    .. math::

        2 \kappa_{lm} = -l \, (l+1) \, \phi_{lm} \;.

    The shear field :math:`\gamma` is related to the deflection potential field
    as

    .. math::

        2 \gamma = \eth\eth \, \phi
        \quad\text{or}\quad
        2 \gamma = \bar{\eth}\bar{\eth} \, \phi \;,

    depending on the definition of the shear field spin weight as :math:`2` or
    :math:`-2`.  In either case, the shear modes :math:`\gamma_{lm}` are
    related to the deflection potential modes as

    .. math::

        2 \gamma_{lm} = \sqrt{(l+2) \, (l+1) \, l \, (l-1)} \, \phi_{lm} \;.

    The shear modes can therefore be obtained via the convergence, or
    directly from the deflection potential.

    '''

    # set to None for the initial iteration
    gamma1, gamma2 = None, None

    while True:
        # return the shear field and wait for the next convergence field
        # break the loop when asked to exit the generator
        try:
            kappa = yield gamma1, gamma2
        except GeneratorExit:
            break

        alm = hp.map2alm(kappa, lmax=lmax, pol=False, use_pixel_weights=True)

        # initialise everything on the first iteration
        if gamma1 is None:
            nside = hp.get_nside(kappa)
            lmax = hp.Alm.getlmax(len(alm))

            log.debug('nside from kappa: %d', nside)
            log.debug('lmax from alms: %s', lmax)

            blm = np.zeros_like(alm)

            l = np.arange(lmax+1)
            fl = np.sqrt((l+2)*(l+1)*l*(l-1))
            fl /= np.clip(l*(l+1), 1, None)
            fl *= -1

        # convert convergence to shear modes
        hp.almxfl(alm, fl, inplace=True)
        gamma1, gamma2 = hp.alm2map_spin([alm, blm], nside, 2, lmax)


@generator('zmin, zmax, delta -> zsrc, kappa')
def convergence(cosmo, weight='midpoint'):
    '''convergence from integrated matter shells'''

    # prefactor
    f = 3*cosmo.Om/2

    # these are the different ways in which the matter can be weighted
    if weight == 'midpoint':

        log.info('will use midpoint lensing weights')

        # consider the lensing weight constant, and integrate the matter
        def w(zi, zj, zk):
            z = cosmo.xc_inv(np.mean(cosmo.xc([zi, zj])))
            x = cosmo.xm(z) if z != 0 else 1.
            return cosmo.xm(z, zk)/cosmo.xm(zk)*(1 + z)*cosmo.vc(zi, zj)/x

    elif weight == 'integrated':

        log.info('will use integrated lensing weights')

        # consider the matter constant, and integrate the lensing weight
        def w(zi, zj, zk):
            z = np.linspace(zi, zj, 100)
            f = cosmo.xm(z)
            f *= cosmo.xm(z, zk)/cosmo.xm(zk)
            f *= (1 + z)/cosmo.e(z)
            return np.trapz(f, z)

    else:
        raise ValueError(f'invalid value for weight: {weight}')

    # initial yield
    z3 = kappa3 = None

    # return convergence and get new matter shell, or stop on exit
    while True:
        try:
            zmin, zmax, delta23 = yield z3, kappa3
        except GeneratorExit:
            break

        # set up variables on first iteration
        if kappa3 is None:
            kappa2 = np.zeros_like(delta23)
            kappa3 = np.zeros_like(delta23)
            delta12 = 0
            z2 = z3 = zmin
            r23 = 1
            w33 = 0

        # deal with non-contiguous redshift intervals
        if z3 != zmin:
            raise NotImplementedError('shells must be contiguous')

        # cycle convergence planes
        # normally: kappa1, kappa2, kappa3 = kappa2, kappa3, <empty>
        # but then we set: kappa3 = (1-t)*kappa1 + ...
        # so we can set kappa3 to previous kappa2 and modify in place
        kappa2, kappa3 = kappa3, kappa2

        # redshifts of source planes
        z1, z2, z3 = z2, z3, zmax

        # extrapolation law
        r12 = r23
        r13, r23 = cosmo.xm([z1, z2], z3)/cosmo.xm(z3)
        t123 = r13/r12

        # weights for the lensing recurrence
        w22 = w33
        w23 = w(z1, z2, z3)
        w33 = w(z2, z3, z3)

        # compute next convergence plane in place of last
        kappa3 *= 1 - t123
        kappa3 += t123*kappa2
        kappa3 += f*(w23 - t123*w22)*delta12
        kappa3 += f*w33*delta23

        # output some statistics
        log.info('zsrc: %f', z3)
        log.info('κbar: %f', np.mean(kappa3))
        log.info('κmin: %f', np.min(kappa3))
        log.info('κmax: %f', np.max(kappa3))
        log.info('κrms: %f', np.sqrt(np.mean(np.square(kappa3))))

        # before losing it, keep current matter slice for next round
        delta12 = delta23


@generator('zmin, zmax, delta, kappa -> kappa')
def ia_nla(cosmo, a_ia, eta=0., z0=0., lbar=0., l0=1e-9, beta=0.):
    r'''Intrinsic alignments using the NLA model

    Parameters
    ----------
    cosmo: :class:`~cosmology._lcdm.LCDM`
        LCDM cosmology model of interest.
    a_ia: float
        Intrinsic alignments amplitude.
    eta: float, optional
        Power of the redshift dependence of the intrinsic alignments amplitude.
        If not given, it is assumed to be 0.
    z0: float, optional
        Reference redshift used in redshift dependence of the intrinsic alignments
        amplitude. If not given, it is assumed to be 0.
    lbar: float, optional
        Mean luminosity of the galaxy sample used in luminosity dependence of the
        intrinsic alignments amplitude. If not given, it is assumed to be 0.
    l0: float, optional
        Reference luminosity used in luminosity dependence of the intrinsic alignments
        amplitude. If not given, it is assumed to be 1e-09.
    beta: float, optional
        Power of the luminosity dependence of the intrinsic alignments amplitude.
        If not given, it is assumed to be 0.

    Receives
    ----------
    zmin: float
        Minimum redshift of the matter shell.
    zmax: float
        Maximum redshift of the matter shell.
    delta: (nside, ) array_like
        Matter overdensity map of the matter shell.
    kappa: (nside, ) array_like
        Convergence map of the matter shell.

    Yields
    ----------
    kappa: (nside, ) array_like
        Updated convergence map of the mattter shell including IA.

    Warnings
    ----------
    This generator changes the kappa maps.

    Notes
    ----------

    The instrinsic alignments convergence :math:`\kappa_{\rm{IA}}` is computed
    from the matter overdensity field :math:`\delta` using the Non-linear
    Alignments Model (NLA) [1]_ [2]_ [3]_ in the following manner.

    We define the intrinsic alignment amplitude :math:`f_{\rm{NLA}}` for a given
    redshift shell at its midpoint in comoving distance at :math:`z` as [4]_

    .. math::

        f_{\rm{NLA}} = - A_{\rm{IA}} \frac{C_{1} \overline{\rho}(z)}{\overline{D}(z)}
        \bigg(\frac{1+z}{1+z_{0}}\bigg)^{\eta}
        \bigg(\frac{\overline{L}}{L_{0}}\bigg)^{\beta}  \;.

    where :math:`A_{\rm{IA}}` is the intrinsic alignments amplitude,
    :math:`C_{1}` is a normalisation constant set to
    :math:`C_{1} = 5 \times 10^{-14} (h^{2} M_{\odot}/Mpc^{-3})^{-2}` [2]_,
    :math:`\overline{\rho}` is the mean matter density of the Universe at redshift :math:`z`,
    :math:`\overline{D}(z)` is the growth factor, :math:`\eta` is the power of the power law
    which describes the redshift-dependence of the IA with respect to :math:`z_{0}` and
    :math:`\beta` is the power of the power law which describes the dependence on luminosity,
    :math:`\overline{L}`, of the IA with respect to :math:`L_{0}`.

    Then, the convergence of the matter shells is updated by adding the contribution from IAs
    to the convergence from lensing as follows

    .. math::

        \kappa(z) = \kappa_{\rm{lensing}}(z) + f_{\rm{NLA}}(z, \overline{L}) \delta(z)

    where :math:`\delta(z)` is the matter overdensity field at redshift :math:`z`.

    References
    ----------
    .. [1] Catelan P., Kamionkowski M., Blandford R. D., 2001, MNRAS, 320, L7. doi:10.1046/j.1365-8711.2001.04105.x
    .. [2] Hirata C. M., Seljak U., 2004, PhRvD, 70, 063526. doi:10.1103/PhysRevD.70.063526
    .. [3] Bridle S., King L., 2007, NJPh, 9, 444. doi:10.1088/1367-2630/9/12/444
    .. [4] Jeffrey N., Alsing J., Lanusse F., 2021, MNRAS, 501, 954. doi:10.1093/mnras/staa3594
    '''

    # initial yield
    kappa_ia = None

    while True:
        try:
            zmin, zmax, delta, kappa = yield kappa_ia
        except GeneratorExit:
            break

        # Get centre of the comoving shell
        dcmin, dcmax = cosmo.dc(np.array([zmin, zmax]))
        dcmid = (dcmax - dcmin)*.5 + dcmin
        zmid = cosmo.dc_inv(dcmid)

        # Get the normalisation constant within the NLA model
        c1 = 5e-14/cosmo.h**2  # Solar masses per cubic Mpc
        rho_c1 = c1*cosmo.rho_c0

        # Apply the NLA model
        prefactor = - a_ia * rho_c1 * cosmo.Om
        inverse_linear_growth = 1. / cosmo.gf(zmid)
        redshift_dependence = ((1+zmid)/(1+z0))**eta
        luminosity_dependence = (lbar/l0)**beta

        f_nla = prefactor * inverse_linear_growth * redshift_dependence * luminosity_dependence

        # Update the convergence field
        kappa_ia = kappa + delta * f_nla


@generator('zsrc, kappa?, gamma1?, gamma2? -> kappa_bar, gamma1_bar, gamma2_bar')
def lensing_dist(z, nz, cosmo):
    '''generate weak lensing maps for source distributions

    This generator takes a single or multiple (via leading axes) redshift
    distribution(s) of sources and computes their integrated mean lensing maps.

    The generator receives the convergence and/or shear maps for each source
    plane.  It then averages the lensing maps by interpolation between source
    planes [1]_, and yields the result up to the last source plane received.

    Parameters
    ----------
    z, nz : array_like
        The redshift distribution(s) of sources.  The redshifts ``z`` must be a
        1D array.  The density of sources ``nz`` must be at least a 1D array,
        with the last axis matching ``z``.  Leading axes define multiple source
        distributions.
    cosmo : Cosmology
        A cosmology instance to obtain distance functions.

    Receives
    --------
    zsrc : float
        Source plane redshift.
    kappa, gamma1, gamma2 : array_like, optional
        HEALPix maps of convergence and/or shear.  Unavailable maps are ignored.

    Yields
    ------
    kappa_bar, gamma1_bar, gamma2_bar : array_like
        Integrated mean lensing maps, or ``None`` where there are no input maps.
        The maps have leading axes matching the distribution.

    References
    ----------
    .. [1] Tessore et al., in prep.

    '''

    # check inputs
    if np.ndim(z) != 1:
        raise TypeError('redshifts must be one-dimensional')
    if np.ndim(nz) == 0:
        raise TypeError('distribution must be at least one-dimensional')
    *sh, sz = np.shape(nz)
    if sz != len(z):
        raise TypeError('redshift axis mismatch')

    # helper function to get normalisation
    # takes the leading distribution axes into account
    def norm(nz_, z_):
        return np.expand_dims(np.trapz(nz_, z_), np.ndim(nz_)-1)

    # normalise distributios
    nz = np.divide(nz, norm(nz, z))

    # total accumulated weight for each distribution
    # shape needs to match leading axes of distributions
    w = np.zeros((*sh, 1))

    # initial lensing plane
    # give small redshift > 0 to work around division by zero
    zsrc = 1e-10
    kap = gam1 = gam2 = 0

    # initial yield
    kap_bar = gam1_bar = gam2_bar = None

    # wait for next source plane and return result, or stop on exit
    while True:
        zsrc_, kap_, gam1_, gam2_ = zsrc, kap, gam1, gam2
        try:
            zsrc, kap, gam1, gam2 = yield kap_bar, gam1_bar, gam2_bar
        except GeneratorExit:
            break

        # integrated maps are initialised to zero on first iteration
        # integrated maps have leading axes for distributions
        if kap is not None and kap_bar is None:
            kap_bar = np.zeros((*sh, np.size(kap)))
        if gam1 is not None and gam1_bar is None:
            gam1_bar = np.zeros((*sh, np.size(gam1)))
        if gam2 is not None and gam2_bar is None:
            gam2_bar = np.zeros((*sh, np.size(gam2)))

        # get the restriction of n(z) to the interval between source planes
        nz_, z_ = restrict_interval(nz, z, zsrc_, zsrc)

        # get integrated weight and update total weight
        # then normalise n(z)
        w_ = norm(nz_, z_)
        w += w_
        nz_ /= w_

        # integrate interpolation factor against source distributions
        t = norm(cosmo.xm(zsrc_, z_)/cosmo.xm(z_)*nz_, z_)
        t /= cosmo.xm(zsrc_, zsrc)/cosmo.xm(zsrc)

        # interpolate convergence planes and integrate over distributions
        for m, m_, m_bar in (kap, kap_, kap_bar), (gam1, gam1_, gam1_bar), (gam2, gam2_, gam2_bar):
            if m is not None:
                m_bar += (t*m + (1-t)*m_ - m_bar)*(w_/w)
