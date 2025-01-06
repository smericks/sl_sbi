# Probabilistic language
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.distributions import constraints
# for app mag. to amplitude conversion
from lenstronomy.Util.data_util import magnitude2cps
from lenstronomy.LightModel.light_model import LightModel


# output AB zeropoint
output_ab_zeropoint = 25.1152 # calculated from an HST .fits header
# TARGETS = (theta_E, gamma1, gamma2, gamma_lens, e1, e2, x_lens, y_lens, x_src, y_src)

def nuisances_prior_model(theta_E, gamma1, gamma2, gamma_lens, e1, e2, 
    x_lens, y_lens, x_src, y_src):
    """This function defines the probabilities of all nuisance simulation parameters,
    conditioned on the target parameters: 
    
    Args:
        theta_E: Einstein radius (")
        gamma1, gamma2: External shear components
        gamma_lens: PEMD Power-law slope
        e1, e2: Ellipticity components
        x_lens,y_lens: Lens mass center coordinates
        x_src,y_src: Quasar center coordinates
    """
    # Parameters of the quasar
    mag_app_quasar = numpyro.sample('quasar_mag_app', fn=dist.LogNormal(3.0, 0.1))
    # in target params
    #cx_quasar = numpyro.sample('quasar_center_x', obs=x_src)
    #cy_quasar = numpyro.sample('quasar_center_y', obs=y_src)

    # Parameters of the source
    mag_app_host = numpyro.sample('source_mag_app', fn=dist.LogNormal(2.0, 0.1))
    R_sersic_host = numpyro.sample('source_R_sersic', fn=dist.TruncatedNormal(0.2, 0.1, low=0.05, high=2.))
    n_sersic_host = numpyro.sample('source_n', fn=dist.Uniform(0.5, 6.))
    e1_host = numpyro.sample('source_e1', fn=dist.TruncatedNormal(0.05, 0.05, low=-0.3, high=0.3))
    e2_host = numpyro.sample('source_e2', fn=dist.TruncatedNormal(0.05, 0.05, low=-0.3, high=0.3))
    # in target params
    #cx_host = cx_quasar
    #cy_host = cy_quasar

    # power-law, with center relative the lens light
    # target params
    #theta_E_lens = numpyro.sample('lens_theta_E', obs=theta_E)
    #gamma_lens = numpyro.sample('lens_gamma', obs=gamma_lens)
    #e1_lens = numpyro.sample('lens_e1', obs=e1)
    #e2_lens = numpyro.sample('lens_e2', obs=e2)
    #center_x_lens = numpyro.sample('lens_center_x', obs=x_lens)
    #center_y_lens = numpyro.sample('lens_center_y', obs=y_lens)

    # external shear
    # target params
    #gamma1_lens = numpyro.sample('lens_gamma1', obs=gamma1)
    #gamma2_lens = numpyro.sample('lens_gamma2', obs=gamma2)

    # Center of the lens light, relative to lens mass
    cx_light = numpyro.sample('light_center_x', fn=dist.Normal(x_lens, 0.005))
    cy_light = numpyro.sample('light_center_y', fn=dist.Normal(y_lens, 0.005))

    # Parameters of the lens light
    mag_app_light = numpyro.sample('light_mag_app', fn=dist.LogNormal(2.0, 0.1))
    R_sersic_light = numpyro.sample('light_R_sersic', fn=dist.Normal(1.0, 0.1)) 
    n_sersic_light = numpyro.sample('light_n', fn=dist.Uniform(2., 5.))
    e1_light = numpyro.sample('light_e1', fn=dist.TruncatedNormal(0.1, 0.05, low=-0.3, high=0.3))
    e2_light = numpyro.sample('light_e2', fn=dist.TruncatedNormal(0.04, 0.05, low=-0.3, high=0.3))

def params2kwargs(nuisance_params,target_params):
    """Function that takes the flattened dictionary of numpyro parameters
    and reshape it back to the argument of lens_image.model() or lens_image.simulation()
    """
    kw = {
    'kwargs_lens': [
        # PEMD
        {'theta_E': target_params['theta_E'],
        'gamma': target_params['gamma_lens'],
        'e1': target_params['e1'],
        'e2': target_params['e2'],
        'center_x': target_params['x_lens'],
        'center_y': target_params['y_lens']},
        # External Shear
        {'gamma1': target_params['gamma1'],
        'gamma2': target_params['gamma2'],
        'ra_0': 0.0,
        'dec_0': 0.0}],

    'kwargs_source': [
        # convert from apparent mag. to amplitude
        {'amp': sersic_mag_to_amp(nuisance_params['source_mag_app'],output_ab_zeropoint,
            {'R_sersic': nuisance_params['source_R_sersic'],
            'n_sersic': nuisance_params['source_n'],
            'e1': nuisance_params['source_e1'],
            'e2': nuisance_params['source_e2'],
            'center_x': target_params['x_src'],
            'center_y': target_params['y_src']}),
        'R_sersic': nuisance_params['source_R_sersic'],
        'n_sersic': nuisance_params['source_n'],
        'e1': nuisance_params['source_e1'],
        'e2': nuisance_params['source_e2'],
        'center_x': target_params['x_src'],
        'center_y': target_params['y_src']}],

    'kwargs_lens_light': [
        # convert from apparent mag. to amplitude
        {'amp': sersic_mag_to_amp(nuisance_params['light_mag_app'],output_ab_zeropoint,
            {'R_sersic': nuisance_params['light_R_sersic'],
            'n_sersic': nuisance_params['light_n'],
            'e1': nuisance_params['light_e1'],
            'e2': nuisance_params['light_e2'],
            'center_x': nuisance_params['light_center_x'],
            'center_y': nuisance_params['light_center_y']}),
        'R_sersic': nuisance_params['light_R_sersic'],
        'n_sersic': nuisance_params['light_n'],
        'e1': nuisance_params['light_e1'],
        'e2': nuisance_params['light_e2'],
        'center_x': nuisance_params['light_center_x'],
        'center_y': nuisance_params['light_center_y']}],

    'kwargs_point_source': [
        {'ra': target_params['x_src'], 
        'dec': target_params['y_src'], 
        # flux = amplitude for point source
        'amp': magnitude2cps(nuisance_params['quasar_mag_app'],
                output_ab_zeropoint)}]
    }
    
    return kw

def sersic_mag_to_amp(mag_apparent,mag_zeropoint,sersic_kwargs):
    """Converts an apparent magnitude to a lenstronomy-defined amplitude
        for a Sersic profile

    Args:
        mag_apparent (float): The desired apparent magnitude
        mag_zeropoint (float): The magnitude zero-point of the detector
        sersic_kwargs (dict): A dict of kwargs for SERSIC_ELLIPSE, amp
            parameter not required

    Returns: 
        (float): amplitude lenstronomy should use to get desired magnitude
    """

    sersic_model = LightModel(['SERSIC_ELLIPSE'])
    # norm=True sets amplitude = 1
    flux_norm = sersic_model.total_flux([sersic_kwargs], norm=True)[0]
    flux_true = magnitude2cps(mag_apparent, mag_zeropoint)
    
    return flux_true/flux_norm
