# TODO: imports
import numpy as np
from lenstronomy.Util import kernel_util
from astropy.io import fits
import jax
# herculens stuff
from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel
from herculens.PointSourceModel.point_source_model import PointSourceModel
from herculens.LensImage.lens_image import LensImage
from herculens.LensImage.Numerics.numerics import Numerics


class LensedAGN():
    """
        Class object that produces simulated lensed AGN using herculens simulator
    """

    def __init__(self,psf_fits_file = 'Simulators/STDPBF_WFC3UV_F814W.fits'):
        """
        Instantiate herculens LensImage() object

        Args:
            psf_fits_file (string): Location of empirical PSF kernel library

        Harcoded assumptions:
            - npix = 80
            - arcsec per pixel = 0.04"
            - supersampling = 3 during raytracing 
            - exposure time = 1400s
            - background rms = 0.0056 e-/s
            - kwargs_lens_eqn_solver = {
                # solutions = 5, 
                # iterations = 5, 
                triangle scale factor = 2
                # triangle subdivisions = 3}
        """
        # Some hardcoded assumptions
        self.npix = 80
        self.arcsec_per_pix = 0.04

        # initialize psf kernels, pixel grid
        self.setup_psf_kernels(psf_fits_file)
        pixel_grid = self.initial_pixel_grid()

        # initialize herculens simulator object
        self.herculens_simulator = LensImage(
            grid_class=pixel_grid, 
            psf_class=PSF(
                psf_type='PIXEL',
                kernel_point_source=self.draw_psf_kernel(),
                kernel_supersampling_factor=1,
            ), 
            noise_class=self.initial_noise_model(),
            lens_mass_model_class=MassModel(['EPL', 'SHEAR']),
            source_model_class=LightModel(['SERSIC_ELLIPSE']),
            lens_light_model_class=LightModel(['SERSIC_ELLIPSE']),
            point_source_model_class=self.initial_ps_model(pixel_grid,
                MassModel(['EPL', 'SHEAR'])),
            kwargs_numerics=dict(supersampling_factor=3),
            kwargs_lens_equation_solver={'nsolutions': 5, 'niter': 5, 
                'scale_factor': 2, 'nsubdivisions': 3})
        
        # TODO: initialize sampling of nuisances
          
    def initial_pixel_grid(self):
        """
        Instantiate herculens PixelGrid() object
        """

        pix_scl = self.arcsec_per_pix 
        half_size = self.npix * pix_scl / 2.
        # position of the (0, 0) with respect to bottom left pixel
        ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2.
        # transformation matrix pixel <-> angle
        transform_pix2angle = pix_scl * np.eye(2) 
        kwargs_pixel = {'nx': self.npix, 'ny': self.npix,
                        'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                        'transform_pix2angle': transform_pix2angle}

        # create the PixelGrid class
        return PixelGrid(**kwargs_pixel)


    def setup_psf_kernels(self,psf_fits_file):
        """
        Load in a library of PSF kernels

        Args:
            psf_fits_file (string): location of focus diverse PSF maps (from: https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/psf)
        """

        with fits.open(psf_fits_file) as hdu:
            psf_kernels = hdu[0].data
        psf_kernels = psf_kernels.reshape(-1,101,101)
        psf_kernels[psf_kernels<0] = 0

        # normalize psf_kernels to sum to 1
        psf_sums = np.sum(psf_kernels,axis=(1,2))
        psf_sums = psf_sums.reshape(-1,1,1)
        normalized_psfs = psf_kernels/psf_sums
        self.psf_kernels = normalized_psfs

    def draw_psf_kernel(self):
        """
        After using setup_psf_kernels() to load in a library of empirical PSFs,
            return a random weighted combination of those PSF kernels, degraded
            by a factor of 4 (for 0.04"/pixel resolution)
        """
        weights = np.random.uniform(size=np.shape(self.psf_kernels)[0])
        weights /= np.sum(weights)
        weighted_sum = np.sum(weights.reshape(len(weights),1,1) * self.psf_kernels,axis=0)
        # downgrade resolution to match observations
        return kernel_util.degrade_kernel(weighted_sum,4)


    def initial_noise_model(self):
        """
        Initialize herculens Noise() object. Assumes 1400s exposure, background
            rms of 0.0056 e-/s
        """
        # exposure time, used for estimating the shot noise
        exp_time = 1400.  # seconds
        # standard deviation of the background noise
        sigma_bkg = 0.0056  # electrons/second (taken from WG0214 F814W header)

        return Noise(
            self.npix, self.npix,
            background_rms=sigma_bkg,
            exposure_time=exp_time,
        )
    
    def initial_ps_model(self,pixel_grid,lens_mass_model):
        """
        Initialize herculens PointSourceModel() object. 

        Args:
            pixel_grid (herculens.Coordinates.pixel_grid.PixelGrid)
            lens_mass_model (herculens.MassModel.mass_model.MassModel)
        """
        # Create a pixel grid for solving the lens equation
        ps_grid = pixel_grid.create_model_grid(pixel_scale_factor=0.5)

        point_source_type_list = ['SOURCE_POSITION']
        return PointSourceModel(point_source_type_list, lens_mass_model, 
            ps_grid)
    
    def __call__(self,target_params):
        """
        TODO: 
        - sample nuisances
        - create kwargs from nuisances + targets
        - simulate & return image
        """

        # Draw nuisance parameters

        # Draw PSF & re-instantiate necessary objects
        self.herculens_simulator.PSF = PSF(
                psf_type='PIXEL',
                kernel_point_source=self.draw_psf_kernel(),
                kernel_supersampling_factor=1,
            )
        self.herculens_simulator.ImageNumerics = Numerics(
            pixel_grid=self.herculens_simulator.Grid,
            psf = self.herculens_simulator.PSF,
            **self.herculens_simulator.kwargs_numerics
        )
        # Create kwargs that are combo of nuisance params and target params
        kwargs_all_input = {}
        # simulate image
        image = self.herculens_simulator.simulation(
            **kwargs_all_input, 
            compute_true_noise_map=True, 
            prng_key=jax.random.PRNGKey(42),
        )

        return image
