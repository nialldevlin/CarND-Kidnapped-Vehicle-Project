/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // TODO: Set the number of particles
  
  normal_distribution<double> dist_x(gps_x, std_x);
  
  // TODO: Create normal distributions for y and theta
  normal_distribution<double> dist_y(gps_y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    
    particle.x = dist_x(this->gen);
    particle.y = dist_y(this->gen);
    particle.theta = dist_theta(this->gen);
    particle.weight = 1;
    this->particles.push_back(particle);
  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  
  for(Particle &particle : this->particles){    
    particle.x += (velocity / yaw_rate) * sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta);
    particle.y += (velocity / yaw_rate) * cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t);
    particle.theta += yaw_rate;
    
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);
    
    particle.x += dist_x(this->gen);
    particle.y += dist_y(this->gen);
    particle.theta += dist_theta(this->gen);
  }

}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for(LandmarkObs &p : predicted){
    double dist = 0;
  	double prev_dist = std::numeric_limits<double>::infinity();;
    for(const LandmarkObs &obs : observations){
      dist = dist(p.x, p.x, obs.x, obs.y);
      if(dist < prev_dist){
        prev_dist = dist;
        p.id = obs.id;
      }
    }
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for(Particle &p : this->particles){
    //Convert each observation coordinates to map
    vector<LandmarkObs> mapObs;
    LandmarkObs obs;
    for(const LandmarkObs &o : observations){
      // transform to map x coordinate
      double x_map;
      obs.x = p.x + (cos(p.theta) * o.x) - (sin(p.theta) * o.y);

      // transform to map y coordinate
      double y_map;
      obs.y = y_part + (sin(theta) * o.x) + (cos(p.theta) * o.y);
      mapObs.push_back(obs);
    }
    vector<LandmarkObs> lm_in_range;
    //Find landmarks in sensor range
    for(const LandmarkObs &l : map_landmarks){
      if(dist(l.x, l.y, p.x, p.y) <= sensor_range){
        lm_in_range.push_back(l);
      }
    }
    //Map observations to landmarks
    dataAssociation(lm_in_range, observations);
    //Calculate weights with gaussian
    // calculate normalization term
    p.weight = 1.;
    for(const LandmarkObs &lm : lm_in_range){
      double gauss_norm;
      gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

      // calculate exponent
      double exponent;
      exponent = (pow(observations[lm.id].x - lm.x, 2) / (2 * pow(std_landmark[0], 2)))
                   + (pow(observations[lm.id].y - lm.y, 2) / (2 * pow(std_landmark[1], 2)));

      // calculate weight using normalization terms and exponent
      p.weight *= gauss_norm * exp(-exponent);
    }
    
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles(this->particles.size());
  index = gen % 
    beta = 0.0
    mw = max(w)
    for i in range(N):
        beta += random.random() * 2.0 * mw
        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % N
        p3.append(p[index])
    p = p3
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}