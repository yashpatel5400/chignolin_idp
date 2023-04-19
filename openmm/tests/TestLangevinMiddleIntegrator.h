/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2020 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/AssertionUtilities.h"
#include "openmm/Context.h"
#include "openmm/CustomExternalForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/LangevinMiddleIntegrator.h"
#include "SimTKOpenMMRealType.h"
#include "sfmt/SFMT.h"
#include <iostream>
#include <vector>

using namespace OpenMM;
using namespace std;

const double TOL = 1e-5;

void testSingleBond() {
    System system;
    system.addParticle(2.0);
    system.addParticle(2.0);
    LangevinMiddleIntegrator integrator(0, 0.1, 0.01);
    HarmonicBondForce* forceField = new HarmonicBondForce();
    forceField->addBond(0, 1, 1.5, 1);
    system.addForce(forceField);
    Context context(system, integrator, platform);
    vector<Vec3> positions(2);
    positions[0] = Vec3(-1, 0, 0);
    positions[1] = Vec3(1, 0, 0);
    context.setPositions(positions);
    
    // This is simply a damped harmonic oscillator, so compare it to the analytical solution.
    
    double freq = std::sqrt(1-0.05*0.05);
    for (int i = 0; i < 1000; ++i) {
        State state = context.getState(State::Positions | State::Velocities);
        double time = state.getTime();
        double expectedDist = 1.5+0.5*std::exp(-0.05*time)*std::cos(freq*time);
        ASSERT_EQUAL_VEC(Vec3(-0.5*expectedDist, 0, 0), state.getPositions()[0], 0.02);
        ASSERT_EQUAL_VEC(Vec3(0.5*expectedDist, 0, 0), state.getPositions()[1], 0.02);
        double expectedSpeed = -0.5*std::exp(-0.05*time)*(0.05*std::cos(freq*time)+freq*std::sin(freq*time));
        ASSERT_EQUAL_VEC(Vec3(-0.5*expectedSpeed, 0, 0), state.getVelocities()[0], 0.02);
        ASSERT_EQUAL_VEC(Vec3(0.5*expectedSpeed, 0, 0), state.getVelocities()[1], 0.02);
        integrator.step(1);
    }
    
    // Now set the friction to 0 and see if it conserves energy.
    
    integrator.setFriction(0.0);
    context.setPositions(positions);
    State state = context.getState(State::Energy);
    double initialEnergy = state.getKineticEnergy()+state.getPotentialEnergy();
    for (int i = 0; i < 1000; ++i) {
        state = context.getState(State::Energy);
        double energy = state.getKineticEnergy()+state.getPotentialEnergy();
        ASSERT_EQUAL_TOL(initialEnergy, energy, 0.01);
        integrator.step(1);
    }
}

void testTemperature() {
    const int numParticles = 8;
    const double temp = 100.0;
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(5, 0, 0), Vec3(0, 5, 0), Vec3(0, 0, 5));
    LangevinMiddleIntegrator integrator(temp, 3.0, 0.01);
    NonbondedForce* forceField = new NonbondedForce();
    forceField->setNonbondedMethod(NonbondedForce::CutoffPeriodic);
    for (int i = 0; i < numParticles; ++i) {
        system.addParticle(2.0);
        forceField->addParticle((i%2 == 0 ? 1.0 : -1.0), 1.0, 5.0);
    }
    system.addForce(forceField);
    Context context(system, integrator, platform);
    vector<Vec3> positions(numParticles);
    for (int i = 0; i < numParticles; ++i)
        positions[i] = Vec3((i%2 == 0 ? 2 : -2), (i%4 < 2 ? 2 : -2), (i < 4 ? 2 : -2));
    context.setPositions(positions);
    
    // Let it equilibrate.
    
    integrator.step(5000);
    
    // Now run it for a while and see if the temperature is correct.
    
    double ke = 0.0;
    int steps = 10000;
    for (int i = 0; i < steps; ++i) {
        State state = context.getState(State::Energy);
        ke += state.getKineticEnergy();
        integrator.step(1);
    }
    ke /= steps;
    double expected = 0.5*numParticles*3*BOLTZ*temp;
    ASSERT_USUALLY_EQUAL_TOL(expected, ke, 6/std::sqrt((double) steps));
}

void testConstraints() {
    const int numParticles = 8;
    const int numConstraints = 5;
    const double temp = 100.0;
    System system;
    LangevinMiddleIntegrator integrator(temp, 2.0, 0.01);
    integrator.setConstraintTolerance(1e-5);
    NonbondedForce* forceField = new NonbondedForce();
    for (int i = 0; i < numParticles; ++i) {
        system.addParticle(10.0);
        forceField->addParticle((i%2 == 0 ? 0.2 : -0.2), 0.5, 5.0);
    }
    system.addConstraint(0, 1, 1.0);
    system.addConstraint(1, 2, 1.0);
    system.addConstraint(2, 3, 1.0);
    system.addConstraint(4, 5, 1.0);
    system.addConstraint(6, 7, 1.0);
    system.addForce(forceField);
    Context context(system, integrator, platform);
    vector<Vec3> positions(numParticles);
    vector<Vec3> velocities(numParticles);
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);

    for (int i = 0; i < numParticles; ++i) {
        positions[i] = Vec3(i/2, (i+1)/2, 0);
        velocities[i] = Vec3(genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5, genrand_real2(sfmt)-0.5);
    }
    context.setPositions(positions);
    context.setVelocities(velocities);

    // Simulate it and see whether the constraints remain satisfied.

    for (int i = 0; i < 1000; ++i) {
        State state = context.getState(State::Positions);
        for (int j = 0; j < numConstraints; ++j) {
            int particle1, particle2;
            double distance;
            system.getConstraintParameters(j, particle1, particle2, distance);
            Vec3 p1 = state.getPositions()[particle1];
            Vec3 p2 = state.getPositions()[particle2];
            double dist = std::sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])+(p1[2]-p2[2])*(p1[2]-p2[2]));
            ASSERT_EQUAL_TOL(distance, dist, 1e-4);
        }
        integrator.step(1);
    }
}

void testConstrainedMasslessParticles() {
    System system;
    system.addParticle(0.0);
    system.addParticle(1.0);
    system.addConstraint(0, 1, 1.5);
    vector<Vec3> positions(2);
    positions[0] = Vec3(-1, 0, 0);
    positions[1] = Vec3(1, 0, 0);
    LangevinMiddleIntegrator integrator(300.0, 2.0, 0.01);
    bool failed = false;
    try {
        // This should throw an exception.
        
        Context context(system, integrator, platform);
    }
    catch (exception& ex) {
        failed = true;
    }
    ASSERT(failed);
    
    // Now make both particles massless, which should work.
    
    system.setParticleMass(1, 0.0);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(300.0);
    integrator.step(1);
    State state = context.getState(State::Velocities);
    ASSERT_EQUAL(0.0, state.getVelocities()[0][0]);
}

void testRandomSeed() {
    const int numParticles = 8;
    const double temp = 100.0;
    System system;
    LangevinMiddleIntegrator integrator(temp, 2.0, 0.01);
    NonbondedForce* forceField = new NonbondedForce();
    for (int i = 0; i < numParticles; ++i) {
        system.addParticle(2.0);
        forceField->addParticle((i%2 == 0 ? 1.0 : -1.0), 1.0, 5.0);
    }
    system.addForce(forceField);
    vector<Vec3> positions(numParticles);
    vector<Vec3> velocities(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        positions[i] = Vec3((i%2 == 0 ? 2 : -2), (i%4 < 2 ? 2 : -2), (i < 4 ? 2 : -2));
        velocities[i] = Vec3(0, 0, 0);
    }

    // Try twice with the same random seed.

    integrator.setRandomNumberSeed(5);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocities(velocities);
    integrator.step(10);
    State state1 = context.getState(State::Positions);
    context.reinitialize();
    context.setPositions(positions);
    context.setVelocities(velocities);
    integrator.step(10);
    State state2 = context.getState(State::Positions);

    // Try twice with a different random seed.

    integrator.setRandomNumberSeed(10);
    context.reinitialize();
    context.setPositions(positions);
    context.setVelocities(velocities);
    integrator.step(10);
    State state3 = context.getState(State::Positions);
    context.reinitialize();
    context.setPositions(positions);
    context.setVelocities(velocities);
    integrator.step(10);
    State state4 = context.getState(State::Positions);

    // Compare the results.

    for (int i = 0; i < numParticles; i++) {
        for (int j = 0; j < 3; j++) {
            ASSERT_EQUAL_TOL(state1.getPositions()[i][j], state2.getPositions()[i][j], 1e-6);
            ASSERT_EQUAL_TOL(state3.getPositions()[i][j], state4.getPositions()[i][j], 1e-6);
            ASSERT(state1.getPositions()[i][j] != state3.getPositions()[i][j]);
        }
    }
}

void testInitialTemperature() {
    // Check temperature initialization for a collection of randomly placed particles
    const int numParticles = 50000;
    const int nDoF = 3 * numParticles;
    const double targetTemperature = 300;
    System system;
    OpenMM_SFMT::SFMT sfmt;
    init_gen_rand(0, sfmt);
    std::vector<Vec3> positions(numParticles);

    for (int i = 0; i < numParticles; i++) {
        system.addParticle(1.0);
        positions[i][0] = genrand_real2(sfmt);
        positions[i][1] = genrand_real2(sfmt);
        positions[i][2] = genrand_real2(sfmt);
    }

    LangevinMiddleIntegrator integrator(300, 25, 0.001);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    context.setVelocitiesToTemperature(targetTemperature);
    auto velocities = context.getState(State::Velocities).getVelocities();
    double kineticEnergy = 0;
    for(const auto &v : velocities) kineticEnergy += 0.5 * v.dot(v);
    double temperature = (2*kineticEnergy / (nDoF*BOLTZ));
    ASSERT_USUALLY_EQUAL_TOL(targetTemperature, temperature, 0.01);
}

void testForceGroups() {
    System system;
    system.addParticle(1.0);
    LangevinMiddleIntegrator integrator(0.0, 1.0, 0.01);
    integrator.setIntegrationForceGroups(1<<1);
    CustomExternalForce* f1 = new CustomExternalForce("x");
    f1->addParticle(0);
    f1->setForceGroup(1);
    CustomExternalForce* f2 = new CustomExternalForce("y");
    f2->addParticle(0);
    f2->setForceGroup(2);
    system.addForce(f1);
    system.addForce(f2);
    Context context(system, integrator, platform);
    context.setPositions(vector<Vec3>(1));

    // Take one step and verify that the position was updated based only on f1.

    integrator.step(1);
    Vec3 pos = context.getState(State::Positions).getPositions()[0];
    ASSERT(pos[0] < 0);
    ASSERT(pos[1] == 0);
    ASSERT(pos[2] == 0);
}

void runPlatformTests();

int main(int argc, char* argv[]) {
    try {
        initializeTests(argc, argv);
        testSingleBond();
        testTemperature();
        testConstraints();
        testConstrainedMasslessParticles();
        testRandomSeed();
        testInitialTemperature();
        testForceGroups();
        runPlatformTests();
    }
    catch(const exception& e) {
        cout << "exception: " << e.what() << endl;
        return 1;
    }
    cout << "Done" << endl;
    return 0;
}
