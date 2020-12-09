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

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  this->num_particles = 200;	//Set number of particles
  
  std::normal_distribution<double> dist_x(x, std[0]);	//X distribution
  std::normal_distribution<double> dist_y(y, std[1]);	//Y distribution
  std::normal_distribution<double> dist_theta(theta, std[2]);	//Theta distribution
  
  this->particles.reserve(num_particles);	//Set size for faster particle loading
  this->weights.reserve(num_particles);		//Set size for faster weight loading

  for (int i = 0; i < this->num_particles; i++) {
    Particle particle;
    
    particle.id = i;	//Particle id is location in the vector
    particle.x = dist_x(this->gen);		//Set x position with random gaussian noise
    particle.y = dist_y(this->gen);		//Set y position with random gaussian noise
    particle.theta = dist_theta(this->gen);		//Set theta position with random gaussian noise
    particle.weight = 1;	//Set weight to 1
    this->weights.push_back(particle.weight);	//Add weights
    this->particles.push_back(particle);		//Add particle
  }
  this->is_initialized = true;	//It is initialized
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {

  std::normal_distribution<double> dist_x(0, std_pos[0]);	//Random x gaussian noise
  std::normal_distribution<double> dist_y(0, std_pos[1]);	//Random y guassian noise
  std::normal_distribution<double> dist_theta(0, std_pos[2]);	//Random theta gaussian noise
  
  for(Particle &particle : this->particles){
    double theta = particle.theta;
    
    if(fabs(yaw_rate) >= 0.00001){	//If yaw_rate not zero use equations
      particle.x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particle.y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t;      
    } else {	//If yaw rate is zero use equations
      particle.x += velocity * delta_t * cos(theta);
      particle.y += velocity * delta_t * sin(theta);
      particle.theta = theta;
    }
    particle.x += dist_x(this->gen);	//Add x noise
    particle.y += dist_y(this->gen);	//Add y noise
    particle.theta += dist_theta(this->gen);	//Add theta noise
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> &predicted, 
                                     const std::vector<LandmarkObs> observations) {
  for(LandmarkObs &p : predicted){
    double distance = 0;
  	double prev_dist = std::numeric_limits<double>::infinity();
    p.id = -1;
    for(unsigned int i = 0; i < observations.size(); i++){
      distance = dist(p.x, p.y, observations[i].x, observations[i].y);
      if(distance < prev_dist){	//Loop until smallest distance, this is the closest
        prev_dist = distance;
        p.id = i;	//Set ID to location in observations vector of corrosponding observation
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {

  for(Particle &p : this->particles){
    
    //Convert each observation coordinates to map
    std::vector<LandmarkObs> mapObs;
    LandmarkObs obs;
    for(const LandmarkObs &o : observations){
      // transform to map x coordinate
      obs.x = p.x + (cos(p.theta) * o.x) - (sin(p.theta) * o.y);

      // transform to map y coordinate
      obs.y = p.y + (sin(p.theta) * o.x) + (cos(p.theta) * o.y);
      mapObs.push_back(obs);
    }
    
    //Find landmarks in sensor range
    std::vector<LandmarkObs> lm_in_range;
    for(const Map::single_landmark_s &l : map_landmarks.landmark_list){
      if(dist(l.x_f, l.y_f, p.x, p.y) <= sensor_range){
        LandmarkObs lm;
        lm.x = l.x_f;
        lm.y = l.y_f;
        lm_in_range.push_back(lm);
      }
    }
    
    //Map observations to landmarks
    dataAssociation(lm_in_range, mapObs);
    std::cout << std:: endl << "Map OBS " << std::flush;
    for (int i = 0; i < mapObs.size(); i++){
      std::cout << i << " X: " << mapObs[i].x << " Y: " << mapObs[i].y << ' ' << std::flush;
    }
    
    std::cout << std:: endl << "Landmarks in range " << std::flush;
    for (auto &i : lm_in_range)
      std::cout << " ID: " << i.id << " X: " << i.x << " Y: " << i.y << ' ' << std::flush;
    //Calculate weights with gaussian
    p.weight = 1.;
    this->weights[p.id] = 1.;
    for(const LandmarkObs &lm : lm_in_range){
      // calculate normalization term
      double gauss_norm;
      gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

      // calculate exponent
      double exponent;
      exponent = (pow(mapObs[lm.id].x - lm.x, 2) / (2 * pow(std_landmark[0], 2)))
               + (pow(mapObs[lm.id].y - lm.y, 2) / (2 * pow(std_landmark[1], 2)));

      // calculate weight using normalization terms and exponent
      p.weight *= gauss_norm * exp(-exponent);
      this->weights[p.id] = p.weight;
    }
    
  }
}

void ParticleFilter::resample() {
  std::vector<Particle> new_particles(this->num_particles);	//New list of particles
  std::discrete_distribution<int> dist(weights.begin(), weights.end());	//Discrete distribution from weights
  for(Particle &p : new_particles){
    int id = dist(this->gen);
    p = particles[id];	//select particles based on id picked from discrete distribution
  }
  this->particles = new_particles;	//replace old particles
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, 
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}