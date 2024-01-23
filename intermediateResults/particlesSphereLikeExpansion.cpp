#include <array>
#include <numbers>
#include <source_location>
#include <string>
#include <vector>

#include <TBrowser.h>
#include <TFile.h>
#include <TH2D.h>
#include <TH3D.h>

#include "../include/Generators/RealNumberGenerator.hpp"
#include "../include/Particles/Particles.hpp"

template <typename T>
std::vector<T> createParticles(size_t count)
{
    RealNumberGenerator rng;
    std::vector<T> particles(count);

    for (size_t i{}; i < count; ++i)
    {

        particles[i] = T(50, 50, 0,
                         0,0,2674.8237375287363);
    }
    return particles;
}
using std::cout;
int main()
{
    std::string root_file{std::source_location::current().file_name()};
    root_file.erase(root_file.find(".cpp"));
    root_file += ".root";

    TFile *file{new TFile(root_file.c_str(), "recreate")};
    if (!file->IsOpen())
    {
        std::cout << std::format("Error: can't open file {}\n", file->GetName());
        return EXIT_FAILURE;
    }

    RealNumberGenerator rng;
    ParticleAluminiumVector p_Al(createParticles<ParticleAluminium>(1'000'000));
    ParticleArgon p_Ar;
    double sigma =4.5616710728287193E-20;
    
    double n_concentration = 3.7406783738272796e+20;

    constexpr int frames{10};
    std::array<TH3D *, frames> snapshots;
    for (int i{}; i < frames; ++i)
        snapshots[i] = new TH3D(Form("volume_snapshot_%d", i), Form("Snapshot %d", i), 50, 0, 100, 50, 0, 100, 50, 0, 100);

    int snapshot_idx{};
    int time_interval(10E-6), time_step(1E-6), cur_time{}; // time in [ms]
    while (cur_time < time_interval)
    {
        for (size_t i{}; i < p_Al.size(); ++i)
        {
            std::cout<<p_Al[i].getX()<<"\t"<< p_Al[i].getY()<<"\t"<< p_Al[i].getZ()<<"\n";

            p_Al[i].updatePosition(time_step);
            double prob =  sigma*p_Al[i].getVelocityModule()*n_concentration*time_step ;
            cout<<prob<<"\n";
            if (rng() <prob )
                p_Al[i].colide(rng(), p_Al[0].getMass(), p_Ar.getMass());
        }

        // Each 100-th iteration - snapshot
        //if (cur_time % 100 == 0)
        //{
            for (size_t i{}; i < p_Al.size(); ++i){
              std::cout<<p_Al[i].getX()<<"\t"<< p_Al[i].getY()<<"\t"<< p_Al[i].getZ()<<"\n";
                snapshots[snapshot_idx]->Fill(p_Al[i].getX(), p_Al[i].getY(), p_Al[i].getZ());}
            ++snapshot_idx;
        //}

        cur_time += time_step;
    }

    for (int i{}; i < frames; ++i)
        snapshots[i]->Write();

    file->Close();

    return EXIT_SUCCESS;
}
