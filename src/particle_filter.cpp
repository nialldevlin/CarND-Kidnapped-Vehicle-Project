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
  std::cout << "Initialized" << std::endl << std::flush;
  this->num_particles = 10;	//Set number of particles
  
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
  std::cout << "Prediction" << std::endl << std::flush;

  std::normal_distribution<double> dist_x(0, std_pos[0]);	//Random x gaussian noise
  std::normal_distribution<double> dist_y(0, std_pos[1]);	//Random y guassian noise
  std::normal_distribution<double> dist_theta(0, std_pos[2]);	//Random theta gaussian noise
  
  for(Particle &particle : this->particles){
    double theta = particle.theta;
    
    if(fabs(yaw_rate) == 0){	//If yaw_rate zero use equations
      particle.x += velocity * delta_t * cos(theta);
      particle.y += velocity * delta_t * sin(theta);
      particle.theta = theta;
           
    } else {	//If yaw rate is not zero use equations
      particle.x += (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particle.y += (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t; 
    }
    particle.x += dist_x(this->gen);	//Add x noise
    particle.y += dist_y(this->gen);	//Add y noise
    particle.theta += dist_theta(this->gen);	//Add theta noise
  }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, 
                                     std::vector<LandmarkObs> &observations, 
                                     Particle &particle) {
  std::vector<int> ass;
  std::vector<double> s_x;
  std::vector<double> s_y;
  
  for(unsigned int i = 0; i < observations.size(); i++){
    double curr_dist = 0;
  	double prev_dist = std::numeric_limits<double>::infinity();
    observations[i].id = -1;
    for(LandmarkObs &p : predicted){
      curr_dist = dist(p.x, p.y, observations[i].x, observations[i].y);
      if(curr_dist < prev_dist){	//Loop until smallest distance, this is the closest
        prev_dist = curr_dist;
        observations[i].id = i;	//Set ID to location in observations vector of corrosponding observation
      }
    }
    ass.push_back(observations[i].id);
    s_x.push_back(observations[i].x);
    s_y.push_back(observations[i].y);
  }
  //SetAssociations(particle, ass, s_x, s_y);
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  std::cout << "Update Weights" << std::endl << std::flush;

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
    dataAssociation(lm_in_range, mapObs, p);
    
    //Calculate weights with gaussian
    p.weight = 1.;
    this->weights[p.id] = 1.;
    std::cout << std::endl << "landmarks in range: " << lm_in_range.size() << std::endl << std::flush;
    for(const LandmarkObs &m : mapObs){   
      // calculate weight
      std::cout << "Obs x: "m.x 
        		<< ", Landmark x: "  << lm_in_range[m.id].x
                << " , Obs Y: " << m.y 
        		<< ", Landmark Y:  " << lm_in_range[m.id].y << std::endl << std::flush;
      double weight = multiv_prob(std_landmark[0], std_landmark[1], m.x, m.y, lm_in_range[m.id].x, lm_in_range[m.id].y);
      if( weight > 0 ){
      	p.weight *= weight;
      }
    }
    this->weights[p.id] = p.weight;
    std::cout << "Weight: " << p.weight << " " << std::flush;
  }
}

void ParticleFilter::resample() {
  std::cout << "Resampled" << std::endl << std::flush;
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