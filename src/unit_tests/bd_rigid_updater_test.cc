/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <iostream>

//! name the boost unit test module
#define BOOST_TEST_MODULE BDRigidUpdaterTests
#include "boost_utf_configure.h"

#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "TwoStepBDNVT.h"
#include "TwoStepBDNVTRigid.h"
#ifdef ENABLE_CUDA
#include "TwoStepBDNVTRigidGPU.h"
#include "TwoStepBDNVTGPU.h"
#endif

#include "IntegratorTwoStep.h"

#include "BoxResizeUpdater.h"

#include "BinnedNeighborList.h"
#include "Initializers.h"
#include "LJForceCompute.h"

#ifdef ENABLE_CUDA
#include "BinnedNeighborListGPU.h"
#include "LJForceComputeGPU.h"
#endif

#include "FENEBondForceCompute.h"

#ifdef ENABLE_CUDA
#include "FENEBondForceComputeGPU.h"
#endif

#include "saruprng.h"
#include <math.h>
#include <time.h>
#include <strstream>
#include <fstream>
#include <vector>

using namespace std;
using namespace boost;

/*! \file nve_rigid_updater_test.cc
    \brief Implements unit tests for NVERigidUpdater
    \ingroup unit_tests
*/


//! Tolerance for floating point comparisons
#ifdef SINGLE_PRECISION
const Scalar tol = Scalar(1e-2);
#else
const Scalar tol = 1e-3;
#endif

struct AtomInfo
{
	int type, localidx, body;
	double mass;
	double x, y, z;
};

struct BondInfo
{
	char type[50];
	int localidxi, localidxj;
	double kappa, R0, sigma, epsilon;
};

struct BuildingBlock
{
	std::vector<AtomInfo> atoms;
	std::vector<BondInfo> bonds;
	double spacing_x, spacing_y, spacing_z;
};

//! Typedef'd TwoStepBDNVTRigid class factory
typedef boost::function<shared_ptr<TwoStepBDNVTRigid> (shared_ptr<SystemDefinition> sysdef, 
                            shared_ptr<ParticleGroup> group, Scalar T, unsigned int seed)> bdnvtup_creator;

void writeRestart(shared_ptr<SystemDefinition> sysdef, unsigned int timestep);

void readRestart(shared_ptr<SystemDefinition> sysdef, char* file_name);

void load_buildingblock_template(char* template_file, BuildingBlock& buildingBlock);

void dump_xyz(shared_ptr<SystemDefinition> sysdef, unsigned int timestep);

void parse(char* line, char*& command, unsigned int& narg, char**& arg);

void bd_updater_lj_tests(bdnvtup_creator bdup_creator, const ExecutionConfiguration& exec_conf)
    {
#ifdef ENABLE_CUDA
    g_gpu_error_checking = true;
#endif
    
    unsigned int nbodies = 800;
    unsigned int nparticlesperbuildingblock = 5;
    unsigned int nbondsperbuildingblock;
    unsigned int body_size = 5;
    unsigned int natomtypes = 2;
    unsigned int nbondtypes = 1;
    
    unsigned int N = nbodies * nparticlesperbuildingblock;
    Scalar box_length = 24.0814;
    shared_ptr<SystemDefinition> sysdef(new SystemDefinition(N, BoxDim(box_length), natomtypes, nbondtypes, 0, 0, 0, exec_conf));
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    
    shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(sysdef, 0, pdata->getN()-1));
    shared_ptr<ParticleGroup> group_all(new ParticleGroup(sysdef, selector_all));
    
    BoxDim box = pdata->getBox();
    
    // setup a simple initial state
    unsigned int ibody = 0;
    unsigned int iparticle = 0;
    Scalar x0 = box.xlo + 0.01;
    Scalar y0 = box.ylo + 0.01;
    Scalar z0 = box.zlo + 0.01;
    Scalar xspacing = 7.0f;
    Scalar yspacing = 1.0f;
    Scalar zspacing = 2.0f;
    
    BuildingBlock buildingBlock;
    load_buildingblock_template("rod.txt", buildingBlock);
    nparticlesperbuildingblock = buildingBlock.atoms.size();
    nbondsperbuildingblock = buildingBlock.bonds.size();


    unsigned int seed = 258719;
    boost::shared_ptr<Saru> random = boost::shared_ptr<Saru>(new Saru(seed));
    Scalar temperature = 1.4;
    Scalar KE = Scalar(0.0);
    Scalar PE = Scalar(0.0);
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    
    // initialize bodies in a cubic lattice with some velocity profile
    for (unsigned int i = 0; i < nbodies; i++)
        {
        for (unsigned int j = 0; j < nparticlesperbuildingblock; j++)
            {
            arrays.x[iparticle] = x0 + buildingBlock.atoms[j].x;
            arrays.y[iparticle] = y0 + buildingBlock.atoms[j].y;
            arrays.z[iparticle] = z0 + buildingBlock.atoms[j].z;

            arrays.vx[iparticle] = random->d();
            arrays.vy[iparticle] = random->d();
            arrays.vz[iparticle] = random->d();
            
            KE += Scalar(0.5) * (arrays.vx[iparticle]*arrays.vx[iparticle] + arrays.vy[iparticle]*arrays.vy[iparticle] + arrays.vz[iparticle]*arrays.vz[iparticle]);
            
            arrays.type[iparticle] = buildingBlock.atoms[j].type;
					
            if (buildingBlock.atoms[j].body > 0)
                arrays.body[iparticle] = ibody;
                        
            unsigned int head = i * nparticlesperbuildingblock;
            for (unsigned int j = 0; j < nbondsperbuildingblock; j++)
				{
                unsigned int particlei = head + buildingBlock.bonds[j].localidxi;
                unsigned int particlej = head + buildingBlock.bonds[j].localidxj;
					
                sysdef->getBondData()->addBond(Bond(0, particlei, particlej));
				}
                                
            iparticle++;
            }
            
        x0 += xspacing;
        if (x0 + xspacing >= box.xhi)
            {
            x0 = box.xlo + 0.01;
            
            y0 += yspacing;
            if (y0 + yspacing >= box.yhi)
                {
                y0 = box.ylo + 0.01;
                
                z0 += zspacing;
                if (z0 + zspacing >= box.zhi)
                    z0 = box.zlo + 0.01;
                }
            }
            
        ibody++;
        }
        
    assert(iparticle == N);
    
    pdata->release();
    
    Scalar deltaT = Scalar(0.005);
    shared_ptr<TwoStepBDNVTRigid> two_step_bdnvt = bdup_creator(sysdef, group_all, temperature, 453034);
        
    shared_ptr<IntegratorTwoStep> bdnvt_up(new IntegratorTwoStep(sysdef, deltaT));
    bdnvt_up->addIntegrationMethod(two_step_bdnvt);

    shared_ptr<BinnedNeighborListGPU> nlist(new BinnedNeighborListGPU(sysdef, Scalar(2.5), Scalar(0.3)));
    shared_ptr<LJForceComputeGPU> fc(new LJForceComputeGPU(sysdef, nlist, Scalar(2.5)));
    
    // setup some values for alpha and sigma
    Scalar epsilon = Scalar(1.0);
    Scalar sigma = Scalar(1.0);
    Scalar alpha_lj = Scalar(1.0);  // alpha = 1.0: LJ
    Scalar lj1 = Scalar(4.0) * epsilon * pow(sigma, Scalar(12.0));
    Scalar lj2 = alpha_lj * Scalar(4.0) * epsilon * pow(sigma, Scalar(6.0));
    
    // specify the force parameters
    fc->setParams(0,0,lj1,lj2, Scalar(1.122));
    fc->setParams(0,1,lj1,lj2, Scalar(1.122));
    fc->setParams(1,1,lj1,lj2, Scalar(1.122));
    
    bdnvt_up->addForceCompute(fc);
    
    shared_ptr<FENEBondForceComputeGPU> fenebond(new FENEBondForceComputeGPU(sysdef));
    fenebond->setParams(0, Scalar(30.0), Scalar(1.5), Scalar(1.0), Scalar(1.122));
    
//    bdnvt_up->addForceCompute(fenebond);
    
    unsigned int nrigid_dof, nnonrigid_dof;
    Scalar current_temp;
    
    shared_ptr<RigidData> rdata = sysdef->getRigidData();
    
    unsigned int start_step = 0;
    unsigned int equil_steps = 5000000;
    unsigned int steps = 20000000;
    unsigned int sampling = 10000;
    unsigned int dump = 1000000;
    
    // Restart if needed
    bool restart = false;
    if (restart == true)
        {
        // Initialize rigid bodies, bonds, etc.
        sysdef->init();
        
        start_step = 40000000;
        char restart_file[100];
        sprintf(restart_file, "restart_%d.txt", start_step);
        readRestart(sysdef, restart_file);
        
        nrigid_dof = rdata->getNumDOF();
        nnonrigid_dof = 3 * (N - body_size * nbodies);
        }
    else
        {
        // Initialize rigid bodies, bonds, etc.
        sysdef->init();
        
        
        nrigid_dof = rdata->getNumDOF();
        nnonrigid_dof = 3 * (N - body_size * nbodies);
        
        // Rescale particle velocities to desired temperature:
        current_temp = 2.0 * KE / (nrigid_dof + nnonrigid_dof);
        Scalar factor = sqrt(temperature / current_temp);
        
        arrays = pdata->acquireReadWrite();
        for (unsigned int j = 0; j < N; j++)
            {
            arrays.vx[j] *= factor;
            arrays.vy[j] *= factor;
            arrays.vz[j] *= factor;
            }
            
        pdata->release();
        }
        
    // Timing
    clock_t start, end;
    double elapsed;
    char file_name[100];
    FILE* fp;
    Scalar Lx, Ly, Lz;
    
    // Box rescaling to target density during mixing
    shared_ptr<VariantLinear> target_L(new VariantLinear());
    Scalar box_target = 20.0;
    unsigned int resize_freq = 10;
    unsigned int nrescale_times = equil_steps / resize_freq;
    for (unsigned int timestep = 0; timestep < nrescale_times; timestep++)
        {
        Scalar delta = (box_target - box_length) / (Scalar)(nrescale_times);
        target_L->setPoint(timestep * resize_freq, box_length + delta * timestep);
        }
        
    shared_ptr<BoxResizeUpdater> box_resize(new BoxResizeUpdater(sysdef, target_L, target_L, target_L));
    
/*
    // Mix with WCA interactions
    cout << "Equilibrating...\n";
    cout << "Number of particles = " << N << "; Number of rigid bodies = " << rdata->getNumBodies() << "\n";
    cout << "Step\tTemp\tPotEng\tKinEng\tTotalE\tBoxSize\tElapsed time\n";
    
    start = clock();
    
    for (unsigned int i = start_step; i <= equil_steps; i++)
        {
        
        bdnvt_up->update(i);
        
        if (i % resize_freq == 0)
            box_resize->update(i);
            
        if (i % sampling == 0)
            {
            end = clock();
            elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;
            
            arrays = pdata->acquireReadWrite();
            KE = Scalar(0.0);
            for (unsigned int j = 0; j < N; j++)
                KE += Scalar(0.5) * (arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
            PE = fc->calcEnergySum();
            
            current_temp = 2.0 * KE / (nrigid_dof + nnonrigid_dof);
            box = pdata->getBox();
            printf("%8d\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\n", i, current_temp, PE / N, KE / N, (PE + KE) / N, box.xhi - box.xlo, elapsed);
            
            if (i % dump == 0)
                {
                sprintf(file_name, "x_t%d.xyz", i);
                fp = fopen(file_name, "w");
                box = pdata->getBox();
                Lx = box.xhi - box.xlo;
                Ly = box.yhi - box.ylo;
                Lz = box.zhi - box.zlo;
                fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
                for (unsigned int j = 0; j < arrays.nparticles; j++)
                    {
                    if (arrays.type[j] == 1)
                        fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[j], arrays.y[j], arrays.z[j]);
                    else
                        fprintf(fp, "C\t%f\t%f\t%f\n", arrays.x[j], arrays.y[j], arrays.z[j]);
                    }
                    
                fclose(fp);
                }
                
            pdata->release();
            
            
            if (i % dump == 0)
                sysdef->writeRestart(i);
                
                
            }
            
        }
        
        
    end = clock();
    elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;
    printf("Elapased time: %f sec or %f TPS\n", elapsed, (double)steps / elapsed);
    
    start_step = equil_steps+1;
    // End of mixing
 */   
    // Production: turn on LJ interactions between rods
    fc->setParams(1,1,lj1,lj2, Scalar(2.5));
    
    cout << "Production...\n";
    cout << "Number of particles = " << N << "; Number of rigid bodies = " << rdata->getNumBodies() << "\n";
    cout << "Step\tTemp\tPotEng\tKinEng\tTotalE\tElapsed time\n";
    
    start = clock();
    
    for (unsigned int i = start_step; i <= start_step + steps; i++)
        {
        
        bdnvt_up->update(i);
        
        if (i % sampling == 0)
            {
            end = clock();
            elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;
            
            arrays = pdata->acquireReadWrite();
            KE = Scalar(0.0);
            for (unsigned int j = 0; j < N; j++)
                KE += Scalar(0.5) * (arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
            PE = fc->calcEnergySum();
            
            current_temp = 2.0 * KE / (nrigid_dof + nnonrigid_dof);
            printf("%8d\t%12.8g\t%12.8g\t%12.8g\t%12.8g\t%12.8g\n", i, current_temp, PE / N, KE / N, (PE + KE) / N, elapsed);
            
            if (i % dump == 0)
                {
                sprintf(file_name, "x_t%d.xyz", i);
                fp = fopen(file_name, "w");
                
                Lx = box.xhi - box.xlo;
                Ly = box.yhi - box.ylo;
                Lz = box.zhi - box.zlo;
                fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
                for (unsigned int j = 0; j < arrays.nparticles; j++)
                    {
                    if (arrays.type[j] == 1)
                        fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[j], arrays.y[j], arrays.z[j]);
                    else
                        fprintf(fp, "C\t%f\t%f\t%f\n", arrays.x[j], arrays.y[j], arrays.z[j]);
                    }
                    
                fclose(fp);
                
                }
                
            pdata->release();
            
            
            if (i % dump == 0)
                writeRestart(sysdef, i);
            }
        }
        
    end = clock();
    elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;
    printf("Elapased time: %f sec or %f TPS\n", elapsed, (double)steps / elapsed);
    
    }

void writeRestart(shared_ptr<SystemDefinition> sysdef, unsigned int timestep)
    {
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<BondData> bond_data = sysdef->getBondData();
    shared_ptr<RigidData> rigid_data = sysdef->getRigidData();
    BoxDim box = pdata->getBox();
    
    char file_name[100];
    sprintf(file_name, "restart_%d.txt", timestep);
    FILE *fp = fopen(file_name, "w");
    
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // Particles
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    fprintf(fp, "%d\n", arrays.nparticles);
    fprintf(fp, "%d\n", pdata->getNTypes());
    fprintf(fp, "%d\n", bond_data->getNBondTypes());
    fprintf(fp, "%f\t%f\t%f\n", Lx, Ly, Lz);
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        fprintf(fp, "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n", arrays.type[i], arrays.body[i],
                arrays.x[i], arrays.y[i], arrays.z[i],
                arrays.vx[i], arrays.vy[i], arrays.vz[i]);
        }
        
    pdata->release();
    
    // Rigid bodies
    unsigned int n_bodies = rigid_data->getNumBodies();
    
    fprintf(fp, "%d\n", n_bodies);
    if (n_bodies <= 0)
        {
        fclose(fp);
        return;
        }
        
        
        {
        ArrayHandle<Scalar> body_mass_handle(rigid_data->getBodyMass(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> body_size_handle(rigid_data->getBodySize(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> moment_inertia_handle(rigid_data->getMomentInertia(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> angvel_handle(rigid_data->getAngVel(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> orientation_handle(rigid_data->getOrientation(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ex_space_handle(rigid_data->getExSpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ey_space_handle(rigid_data->getEySpace(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> ez_space_handle(rigid_data->getEzSpace(), access_location::host, access_mode::read);
        
        ArrayHandle<Scalar4> particle_pos_handle(rigid_data->getParticlePos(), access_location::host, access_mode::read);
        unsigned int particle_pos_pitch = rigid_data->getParticlePos().getPitch();
        
        for (unsigned int body = 0; body < n_bodies; body++)
            {
            fprintf(fp, "%f\t%f\t%f\n", moment_inertia_handle.data[body].x, moment_inertia_handle.data[body].y, moment_inertia_handle.data[body].z);
            fprintf(fp, "%f\t%f\t%f\n", com_handle.data[body].x, com_handle.data[body].y, com_handle.data[body].z);
            
            fprintf(fp, "%f\t%f\t%f\t%f\n", orientation_handle.data[body].x, orientation_handle.data[body].y, orientation_handle.data[body].z, orientation_handle.data[body].w);
            fprintf(fp, "%f\t%f\t%f\n", ex_space_handle.data[body].x, ex_space_handle.data[body].y, ex_space_handle.data[body].z);
            fprintf(fp, "%f\t%f\t%f\n", ey_space_handle.data[body].x, ey_space_handle.data[body].y, ey_space_handle.data[body].z);
            fprintf(fp, "%f\t%f\t%f\n", ez_space_handle.data[body].x, ez_space_handle.data[body].y, ez_space_handle.data[body].z);
            
            unsigned int len = body_size_handle.data[body];
            for (unsigned int j = 0; j < len; j++)
                {
                unsigned int localidx = body * particle_pos_pitch + j;
                fprintf(fp, "%f\t%f\t%f\n", particle_pos_handle.data[localidx].x, particle_pos_handle.data[localidx].y, particle_pos_handle.data[localidx].z);
                }
            }
            
        }
        
    fclose(fp);
    
    }

void readRestart(shared_ptr<SystemDefinition> sysdef, char* file_name)
    {
    printf("Reading restart file..\n");
    unsigned int nparticles, natomtypes, nbondtypes;
    
    shared_ptr<ParticleData> pdata = sysdef->getParticleData();
    shared_ptr<BondData> bond_data = sysdef->getBondData();
    shared_ptr<RigidData> rigid_data = sysdef->getRigidData();
    
    ParticleDataArrays arrays = pdata->acquireReadWrite();
    
    FILE *fp = fopen(file_name, "r");
    fscanf(fp, "%d\n", &nparticles);
    fscanf(fp, "%d\n", &natomtypes);
    fscanf(fp, "%d\n", &nbondtypes);
    
    if (nparticles != arrays.nparticles || natomtypes != pdata->getNTypes() || nbondtypes != bond_data->getNBondTypes())
        {
        printf("Restart file does not match!\n");
        return;
        }
        
    double Lx, Ly, Lz;
    BoxDim box;
    fscanf(fp, "%lf\t%lf\t%lf\n", &Lx, &Ly, &Lz);
    box.xlo = -0.5 * Lx;
    box.xhi = 0.5 * Lx;
    box.ylo = -0.5 * Ly;
    box.yhi = 0.5 * Ly;
    box.zlo = -0.5 * Lz;
    box.zhi = 0.5 * Lz;
    
    pdata->setBox(box);
    for (unsigned int i = 0; i < nparticles; i++)
        {
        fscanf(fp, "%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\n", &arrays.type[i], &arrays.body[i],
               &arrays.x[i], &arrays.y[i], &arrays.z[i],
               &arrays.vx[i], &arrays.vy[i], &arrays.vz[i]);
        }
        
    pdata->release();
    
    // Rigid bodies
    unsigned int n_bodies = rigid_data->getNumBodies();
    fscanf(fp, "%d\n", &n_bodies);
    if (n_bodies <= 0)
        {
        fclose(fp);
        return;
        }
        
        {
        ArrayHandle<unsigned int> body_size_handle(rigid_data->getBodySize(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> moment_inertia_handle(rigid_data->getMomentInertia(), access_location::host, access_mode::readwrite);
        
        ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> orientation_handle(rigid_data->getOrientation(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> ex_space_handle(rigid_data->getExSpace(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> ey_space_handle(rigid_data->getEySpace(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> ez_space_handle(rigid_data->getEzSpace(), access_location::host, access_mode::readwrite);
        
        ArrayHandle<Scalar4> particle_pos_handle(rigid_data->getParticlePos(), access_location::host, access_mode::readwrite);
        unsigned int particle_pos_pitch = rigid_data->getParticlePos().getPitch();
        
        for (unsigned int body = 0; body < n_bodies; body++)
            {
            fscanf(fp, "%f\t%f\t%f\n", &moment_inertia_handle.data[body].x, &moment_inertia_handle.data[body].y, &moment_inertia_handle.data[body].z);
            fscanf(fp, "%f\t%f\t%f\n", &com_handle.data[body].x, &com_handle.data[body].y, &com_handle.data[body].z);
            
            fscanf(fp, "%f\t%f\t%f\t%f\n", &orientation_handle.data[body].x, &orientation_handle.data[body].y, &orientation_handle.data[body].z, &orientation_handle.data[body].w);
            fscanf(fp, "%f\t%f\t%f\n", &ex_space_handle.data[body].x, &ex_space_handle.data[body].y, &ex_space_handle.data[body].z);
            fscanf(fp, "%f\t%f\t%f\n", &ey_space_handle.data[body].x, &ey_space_handle.data[body].y, &ey_space_handle.data[body].z);
            fscanf(fp, "%f\t%f\t%f\n", &ez_space_handle.data[body].x, &ez_space_handle.data[body].y, &ez_space_handle.data[body].z);
            
            unsigned int len = body_size_handle.data[body];
            for (unsigned int j = 0; j < len; j++)
                {
                unsigned int localidx = body * particle_pos_pitch + j;
                fscanf(fp, "%f\t%f\t%f\n", &particle_pos_handle.data[localidx].x, &particle_pos_handle.data[localidx].y, &particle_pos_handle.data[localidx].z);
                }
                
            }
            
        }
        
    fclose(fp);
    
    }

void load_buildingblock_template(char* template_file, BuildingBlock& buildingBlock)
    {
	FILE* fp = fopen(template_file, "r");
	
	unsigned int number;
	char* entry = new char [256];
	
	// obtain the total number of beads in the building block
	fscanf(fp, "%s\t%lf\t%lf\t%lf\n", entry, &buildingBlock.spacing_x, &buildingBlock.spacing_y, &buildingBlock.spacing_z);
	
	while (!feof(fp))
	{
		fscanf(fp, "%s\t%d\n", entry, &number);
				
		if (strcmp(entry, "[Rigid]") == 0 || strcmp(entry, "[Flexible]") == 0)
		{
			
			for (unsigned int i = 0; i < number; i++)
			{
				AtomInfo atomi;

				fscanf(fp, "%d\t%lf\t%lf\t%lf\t%d\t%d\t%lf\n", &atomi.localidx, &atomi.x, &atomi.y, &atomi.z,
					&atomi.body, &atomi.type, &atomi.mass);
				buildingBlock.atoms.push_back(atomi);
			}
		}
		else if (strcmp(entry, "[Connections]") == 0) 
		{
			
			for (unsigned int i = 0; i < number; i++)
			{
				BondInfo bondi;
				
				fscanf(fp, "%d\t%d\t%s\n", &bondi.localidxi, &bondi.localidxj, bondi.type);
				buildingBlock.bonds.push_back(bondi);
			}
		}
	}

	
	fclose(fp);

    }

void dump_xyz(shared_ptr<SystemDefinition> sysdef, unsigned int timestep)
    {
	shared_ptr<ParticleData> pdata = sysdef->getParticleData();
	ParticleDataArrays arrays = pdata->acquireReadWrite();
	BoxDim box = pdata->getBox();

	char file_name[100];
	FILE* fp;
	Scalar Lx, Ly, Lz;

	sprintf(file_name, "x_t%d.xyz", timestep);
	fp = fopen(file_name, "w");
	box = pdata->getBox();								
	Lx = box.xhi - box.xlo;
	Ly = box.yhi - box.ylo;
	Lz = box.zhi - box.zlo;
	fprintf(fp, "%d\n%f\t%f\t%f\n", arrays.nparticles, Lx, Ly, Lz);
	for (unsigned int j = 0; j < arrays.nparticles; j++)
	{
		if (arrays.type[j] == 1)
			fprintf(fp, "N\t%f\t%f\t%f\n", arrays.x[j], arrays.y[j], arrays.z[j]);
		else
			fprintf(fp, "C\t%f\t%f\t%f\n", arrays.x[j], arrays.y[j], arrays.z[j]);
	}
	
	pdata->release();
	fclose(fp);
    }

void parse(char* line, char*& command, unsigned int& narg, char**& arg)
    {
	if (strlen(line) == 0) return;

	unsigned int i, maxarg = 32;
	arg = new char* [maxarg];
	for (i=0; i<maxarg; i++)
		arg[i] = new char [64];

	char* copy = new char [512];
	strcpy(copy, line);

	// strip any # comment by resetting string terminator
	
	int level = 0;
	char *ptr = copy;
	while (*ptr) 
	{
		if (*ptr == '#' && level == 0) 
		{
			*ptr = '\0';
			break;
		}
	
		ptr++;
	}
	
	// command = 1st arg
	
	command = strtok(line, " \t\n\r\f");
	if (command == NULL) return;
	
	// point arg[] at each subsequent arg
	// treat text between double quotes as one arg
	// insert string terminators in copy to delimit args
	
	narg = 0;
	while (1) 
	{
		arg[narg] = strtok(NULL, " \t\n\r\f");
		if (arg[narg]) 
			narg++;
		else 
			break;
	}

	delete [] copy;
    }


#ifdef ENABLE_CUDA
//! TwoStepBDNVTRigidGPU factory for the unit tests
shared_ptr<TwoStepBDNVTRigid> gpu_bdnvt_creator(shared_ptr<SystemDefinition> sysdef, shared_ptr<ParticleGroup> group, Scalar T, unsigned int seed)
    {
    shared_ptr<VariantConst> T_variant(new VariantConst(T));
    return shared_ptr<TwoStepBDNVTRigid>(new TwoStepBDNVTRigidGPU(sysdef, group, T_variant, seed, false));
    }
#endif

#ifdef ENABLE_CUDA

//! extended LJ-liquid test for the GPU class
BOOST_AUTO_TEST_CASE( BDRigidGPU_rod_tests )
    {
    bdnvtup_creator bdnvt_creator_gpu = bind(gpu_bdnvt_creator, _1, _2, _3, _4);
    bd_updater_lj_tests(bdnvt_creator_gpu, ExecutionConfiguration());
    }
#endif

#ifdef WIN32
#pragma warning( pop )
#endif

