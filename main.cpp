#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "matplotlibcpp.h"
#include <math.h>
#include <iostream>
#include <vector>
#include <random> 
#include <functional>
#include <string.h>

using namespace cv;
using namespace std;
namespace plt = matplotlibcpp;


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


void check_params(Params p)
{
	printf("CAMERA CALCULATED VALUES\n");
	printf("  f-number = \t\t %.2f\n", p.fn);
	printf("  p, q, f (cm) = \t %.1f, %.1f, %.1f\n", p.p*100, p.q*100, p.f*100);
	printf("  M = \t\t\t %.2f\n\n", p.M);
	
	printf("APPROPIATE PIXEL SIZE WITH RESPECT TO D_DIFF (D_DIFF > 3*PIXEL_WIDTH)\n");
	double d_diff = 2.44 * (p.M+1) * p.lambda_ * p.fn;  // diffusion diameter on screen (smallest trace on the screen);
	printf ("  pixel size = \t\t\t\t %f um\n", (p.w_px*1e6));
	printf ("  d_diff on screen = \t\t\t %f um\n\n", (d_diff*1e6));
		
	printf("APPROPIATE PARTICLE SIZING\n");
	printf("  d_diff/M (critical particle size) = \t %f um\n\n", (d_diff/p.M*1e6));

	printf("CONTROL VOLUME MUST BE BIGGER THAN VIEWPORT (w_cv > w_viewport)\n");
	printf("  viewport width, height (cm) = \t %f, %f\n", p.w_vp*100, p.h_vp*100);
	printf("  cv width, height (cm) = \t %f, %f\n\n\n", p.delta_x_cv*100, p.delta_y_cv*100 );	
}


struct Particles
{
	vector<double> X, Y, Z, D;
};


Particles generate_random_particles( int n, double mean, double std, double min, double max,
									 double x0, double y0, double z0, double x1, double y1, double z1 )
{
	Particles p;

	p.D = vector<double>(n);
	p.X = vector<double>(n);
	p.Y = vector<double>(n);
	p.Z = vector<double>(n);

	default_random_engine generator;
	normal_distribution<double> D_dist(mean, std);
	uniform_real_distribution<double> X_dist(x0, x1);
	uniform_real_distribution<double> Y_dist(y0, y1);
	uniform_real_distribution<double> Z_dist(z0, z1);

	for (int i= 0; i <n; i++)
	{
		double d = D_dist(generator);
		if (d < min)
			d = min;
		else if (d > max)
			d = max;
		p.D[i] = d;
		p.X[i] = X_dist(generator);
		p.Y[i] = Y_dist(generator);
		p.Z[i] = Z_dist(generator);
	}
	return p;
}


void check_particles(Params p, Particles particles_1, Particles particles_2)
{
	plt::hist(particles_1.D);
	plt::hist(particles_2.D);
	plt::xlabel("diameter range (m)");
	plt::ylabel("number of particles");
	plt::title("Distribution histogram");
	double d_diff = 2.44 * (p.M+1) * p.lambda_ * p.fn;  // diffusion diameter on screen (smallest trace on the screen);
	plt::axvline(d_diff/p.M);
	plt::show();
}


// TRANSFORM FUNCTIONS
// 		Control Volume space: 3D space of experiment (unit: m)
//		Screen Space: 2D space of screen sensor (unit: m)
//      Pixel Space: 2D space of pixels (unit: px)
//      Mesh Space: 2D space of sub-pixel mesh (unit: px)
struct VECTOR_2D_d
{
	double x;
	double y;
};


struct VECTOR_2D_i{	
	int x;
	int y;
};


VECTOR_2D_d location_on_sensor(Params& p, double X_on_cv, double Y_on_cv, double Z_on_cv)
{
	VECTOR_2D_d location;
	location.x = X_on_cv * p.M;
	location.y = Y_on_cv * p.M;
	return location;
}


VECTOR_2D_d location_on_mesh(Params& p, VECTOR_2D_d location_on_sensor )
{
	VECTOR_2D_d location;
	location.x = p.nx_mesh * (location_on_sensor.x/p.w_screen + 0.5);
	location.y = p.ny_mesh * (location_on_sensor.y/p.h_screen + 0.5);
	return location;
}


void plot_particle(Mat perfect_img, Params& p, double D, double X, double Y, double Z)
{
	auto location = location_on_mesh(p, location_on_sensor(p, X, Y, Z));
	Point center( int(location.x), int(location.y) );
	double radius = round ( (p.M * D/2) * (p.nx_mesh / p.w_screen) ); 
	circle( perfect_img, center, radius, 255, FILLED);
}


Particles move_particles(Particles p0, double dt, VelocityField *V)
{
	for (int i=0; i<p0.D.size(); i++)
	{
		p0.X[i] += V->u(p0.X[i], p0.Y[i], p0.Z[i])*dt;
		p0.Y[i] += V->v(p0.X[i], p0.Y[i], p0.Z[i])*dt;
		p0.Z[i] += V->w(p0.X[i], p0.Y[i], p0.Z[i])*dt;		
	}
	return p0;
}


Mat generate_image (Params p, Particles particles_1, Particles particles_2)
{
	Mat perfect_img(p.nx_mesh, p.ny_mesh, CV_8UC1, 1 );
    for (int i =0; i < p.n_p1; i++)
    {
		plot_particle( perfect_img, p, particles_1.D[i], particles_1.X[i], particles_1.Y[i],particles_1.Z[i]);
	}    
    for (int i =0; i < p.n_p2; i++)
    {
		plot_particle( perfect_img, p, particles_2.D[i], particles_2.X[i], particles_2.Y[i],particles_2.Z[i]);
	}
	Mat processed_img(p.nx_mesh, p.ny_mesh, CV_8UC1, 1 );
	GaussianBlur( perfect_img, processed_img, Size(0,0), p.sigma_mesh );
	Mat resized_img(p.nx_px, p.ny_px,  CV_8UC1, 1 );
	resize(processed_img, resized_img, resized_img.size(), 0, 0, INTER_AREA );
	return resized_img; 
}


int main()
{    
    Params p;
    check_params(p);
    Particles particles_1_A = generate_random_particles(p.n_p1, p.d_p1_mean, p.d_p1_std, p.d_p1_min, p.d_p1_max,
											p.x0_cv, p.y0_cv, p.z0_cv, p.x1_cv, p.y1_cv, p.z1_cv);
	Particles particles_2_A = generate_random_particles(p.n_p2, p.d_p2_mean, p.d_p2_std, p.d_p2_min, p.d_p2_max,
											p.x0_cv, p.y0_cv, p.z0_cv, p.x1_cv, p.y1_cv, p.z1_cv);
	check_particles(p, particles_1_A, particles_2_A );
	      
	Mat image_A = generate_image(p, particles_1_A, particles_2_A );
		
    imwrite("image_A.jpg", image_A ); 
    
    Particles particles_1_B = move_particles(particles_1_A, p.dt,  &(p.velocity_field1)  ) ;
    Particles particles_2_B = move_particles(particles_2_A, p.dt,  &(p.velocity_field2)  ) ; 
    
    Mat image_B = generate_image(p, particles_1_B, particles_2_B );
		
    imwrite("image_B.jpg", image_B ); 
    
    return 0;
}
