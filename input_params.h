#include <math.h>
using namespace std;

class VelocityField {
	public:
	virtual double u (double x, double y, double z) = 0;
	virtual double v (double x, double y, double z) = 0;
	virtual double w (double x, double y, double z) = 0;	
};


class SampleVelocityField1 : public VelocityField {
	public:
	double u(double x, double y, double z) override
	{
		return -y*3.0;
	}
	double v(double x, double y, double z) override
	{
		return x*3.0;
	}
	double w(double x, double y, double z) override
	{
		return 0;
	}
};


class SampleVelocityField2 : public VelocityField {
	public:
	double u(double x, double y, double z) override
	{
		return -y*3.0;
	}
	double v(double x, double y, double z) override
	{
		return x*3.0;
	}
	double w(double x, double y, double z) override
	{
		return 0;
	}
};


struct Params
{
	// -------- CAMERA LENS -----------
	double f = 20e-2;            // focal length 
	double p = 70e-2;            // distance between objects and lens 
	double q = 1 / ((1/f)-(1/p));  // distance between screen and lens 
	double M = q / p;             // magnification ratio 
	double d_a = 10e-3;           // apreture diameter  
	double fn = f / d_a;          // f number 

	// -------- CAMERA SENSOR ----------
	int nx_px = 2048;              // number of pixels columns
	int ny_px = nx_px;          // number of pixel rows (assuming square sensor screen)
	double w_px = 10e-6;            // pixel width 
	double h_px = w_px ;           // pixel height (assuming square pixel)
	double w_screen = w_px * nx_px; // screen width 
	double h_screen = h_px * ny_px; // screen height 
	double iso = 1;                // sensor sensivity (it is just a factor and does not match actual ISO definition)

	// -------- CONTROL VOLUME ---------
	double x0_cv = -2.6e-2;
	double x1_cv = +2.6e-2;
	double y0_cv = -2.6e-2;
	double y1_cv = +2.6e-2;
	double z0_cv = 0;
	double z1_cv = 2e-2;

	double delta_x_cv = x1_cv - x0_cv;
	double delta_y_cv = y1_cv - y0_cv;
	double delta_z_cv = z1_cv - z0_cv;

	// -------- VIEWPORT ----------------
	double w_vp = w_screen / M;
	double h_vp = h_screen / M;

	// -------- LASER -----------------
	double lambda_ = 500e-9;         // laser wavelength
	double p_L = 90;                 // laser power (watts)
	// laser power flux profile
	double phi_L (double x, double y, double z)
	{
		 return p_L / ( delta_y_cv * delta_z_cv );
	}

	// -------- PARTICLES -------------
	int n_p1 = 1000;                // number of group 1 particles
	int n_p2 = 10000;               // number of group 2 particles

	double d_p1_mean = 200e-6;        // particle group 1 mean diameter 
	double d_p1_std = 50e-6;          // particle group 1 diameter standard deviation
	double d_p1_min = 50e-6;          // particle group 1 minimun diameter
	double d_p1_max = 400e-6;         // particle group 1 maximum diameter

	double d_p2_mean = 50e-6;          // particle group 2 mean diameter 
	double d_p2_std = 8e-6;         // particle group 2 diameter standard deviation
	double d_p2_min = 20e-6;         // particle group 2 minimun diameter
	double d_p2_max = 70e-6;         // particle group 2 maximum diameter"
	
	// ---------MESH ------------------
	double smallest_particle_diameter = min( d_p1_min, d_p2_min);
	int pixel_subdivisions = ceil (5* w_px / smallest_particle_diameter);
	int nx_mesh = pixel_subdivisions * nx_px;
	int ny_mesh = pixel_subdivisions * ny_px;
	
	// --------PSF ---------------------
	double sigma = 1.414 * (M+1) * lambda_ * fn;  
	double sigma_mesh = (sigma / w_px) * pixel_subdivisions;
	
	// -------VELOCITY_FIELD -----------
	double dt = 0.01;
	SampleVelocityField1 velocity_field1;
	SampleVelocityField2 velocity_field2;
};
