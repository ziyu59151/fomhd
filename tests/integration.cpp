#include "support.hpp"
#include "fomhd/verification.hpp"
#include <algorithm>
#include <bit>
#include <iostream>
#include <numbers>
#include <set>
#include <string>
using namespace fomhd;
void require(bool good,const char* text) {if(!good)throw std::runtime_error(text);}
std::string check_identity(const Check& x) {
    return x.name+"|"+x.population+"|"+x.component+"|"+x.producer+"|"+x.region+"|"+
      std::to_string(static_cast<int>(x.norm))+"|"+x.tag.stage+"|"+
      std::to_string(x.tag.step)+"|"+std::to_string(x.tag.generation)+"|"+
      std::to_string(x.tag.trial)+"|"+std::to_string(x.tag.spectral_context)+"|"+
      std::to_string(std::bit_cast<std::uint64_t>(x.tag.time));
}
bool has_check(const std::vector<Check>& checks, const std::string& name,
               const std::string& stage, bool unavailable=false) {
    return std::any_of(checks.begin(),checks.end(),[&](const Check& x) {
        return x.name==name&&x.tag.stage==stage&&(!unavailable||!x.available);
    });
}
bool has_array(const std::map<std::string,Vec>& arrays, const std::string& prefix) {
    for(const auto& key : {"B","n_i","momentum_i","p_i","n_e","momentum_e","p_e"})
        if(!arrays.contains(prefix+"_"+key))return false;
    return arrays.contains(prefix+"_p_parallel_e");
}
void require_no_duplicate_checks(const std::vector<Check>& checks) {
    std::set<std::string> keys;
    for(const auto& x : checks)
        require(keys.insert(check_identity(x)).second,"duplicate failure check key");
}
class SyncFailureCompute final : public Compute {
  public:
    void synchronize() override {
        if(fail)throw NumericalError("injected pre-commit synchronization failure");
    }
    bool fail{};
};
int main(int argc,char**argv) {
 try {
    const std::string which=argc>1?argv[1]:"equilibrium";
    auto p=test::parameters();Compute cpu;SyncFailureCompute sync_compute;
    Compute& active=which=="sync_failure"?static_cast<Compute&>(sync_compute):
                                         static_cast<Compute&>(cpu);
    auto g=build_rectangle(4,4,1,0,.2,.2);auto op=build_operators(g,p.K);
    Context c{p,g,op,active,p.Nphi};
    auto initial=closed_reference(c,p.ref.ni,p.ref.pressure,p.ref.pressure,p.ref.magnetic,
                                  which=="evolution"?.01:0);
    if(which=="sync_failure") {
        Cumulative ledger;StageResult frame;
        auto committed=initialize(initial,c,ledger,false,&frame);
        auto committed_state=committed.state;
        auto cumulative_before=ledger;
        sync_compute.fail=true;
        bool rejected=false;
        try{(void)advance(committed,ledger,c);}
        catch(const Failure& e){
            rejected=true;
            require(e.code=="NumericalLibrary","sync failure code");
            require(e.stage.has_value()&&e.stage->stage=="interval",
                    "sync failure lost interval identity");
            for(const auto& stage : {"base","midpoint","endpoint"})
                require(has_array(e.partial_arrays,stage),
                        "sync failure lost stage state arrays");
            for(const auto& stage : {"base","midpoint","endpoint"})
                require(has_check(e.checks,"minimum_n",stage),
                        "sync failure lost stage diagnostics");
            require(has_check(e.checks,"CT_update","interval"),
                    "sync failure lost completed interval checks");
            require_no_duplicate_checks(e.checks);
        }
        require(rejected,"sync failure was not converted to Failure");
        require(committed.state.B==committed_state.B&&
                committed.state.fluid[0].n==committed_state.fluid[0].n&&
                committed.state.step==committed_state.step&&
                ledger.accepted_intervals==cumulative_before.accepted_intervals,
                "sync failure changed committed state or ledger");
        std::cout<<"PASS sync_failure\n";return 0;
    }
    if(which=="initialization_gate") {
        p.budgets.at("initial_ampere.linf").atol=-1;
        Cumulative rejected_ledger;
        bool rejected=false;
        try{(void)initialize(initial,c,rejected_ledger,false);}
        catch(const Failure& e){
            rejected=true;
            require(e.code=="HardGate","initialization failure code");
            require(e.stage.has_value()&&e.stage->stage=="initial",
                    "initialization failure lost stage identity");
            require(has_array(e.partial_arrays,"trial"),
                    "initialization failure lost state arrays");
            require(has_check(e.checks,"initial_ampere","initial"),
                    "initialization failure lost completed checks");
            require_no_duplicate_checks(e.checks);
        }
        require(rejected,"initialization gate did not reject");
        std::cout<<"PASS initialization_gate\n";return 0;
    }
    if(which=="failure_evidence") {
        initial.fluid[0].perp[p.K*op.Nc]=-1;
        bool captured=false;try{(void)evaluate_stage(initial,{0,initial.generation,1,0,initial.time,"initial"},c);}
        catch(const Failure& e){
            captured=e.stage.has_value()&&e.partial_arrays.contains("trial_n_i")&&
              e.partial_arrays.contains("trial_p_i")&&!e.checks.empty();
            require_no_duplicate_checks(e.checks);
        }
        require(captured,"rejected stage lost partial state or unavailable diagnostic records");
        std::cout<<"PASS failure_evidence\n";return 0;
    }
    Cumulative ledger;StageResult frame;
    auto u=initialize(initial,c,ledger,which=="equilibrium",&frame);
    require(!frame.checks.empty()&&!frame.fJ0.empty(),"missing initialized diagnostics/current response");
    if(which=="pressure_ledger") {
        double volume=0;for(double w:op.M0)volume+=2*std::numbers::pi*w;
        for(int a=0;a<2;++a){
            auto& f=frame.fluid[a];std::fill(f.theta.begin(),f.theta.end(),2.0);
            std::fill(f.strain.begin(),f.strain.end(),.5);
            std::fill(f.pdot_perp.begin(),f.pdot_perp.end(),3e-9);
            frame.rhs.fluid[a].perp=forward(f.pdot_perp,op.Nc,c);
            if(a){std::fill(f.pdot_parallel.begin(),f.pdot_parallel.end(),5e-9);frame.rhs.fluid[a].parallel=forward(f.pdot_parallel,op.Nc,c);}
            auto rate=power_ledger(frame,a,c);
            const double expected_perp=volume*p.ref.pressure*(a?-1.5:-2.0);
            require(std::abs(rate.perp_rate-expected_perp)<1e-20,"internal source ledger uses pressure derivative instead of internal energy source");
            require(std::abs(rate.spectral_perp)<1e-20,"spectral pressure ledger uses inconsistent energy factors");
            if(a)require(std::abs(rate.parallel_rate+volume*p.ref.pressure*.5)<1e-20,"parallel internal energy requires one-half pressure rate");
        }
        std::cout<<"PASS pressure_ledger\n";return 0;
    }
    auto before=u.state;
    if(which=="rollback") {
        p.budgets.at("CT_update.linf").atol=-1;
        bool rejected=false;
        try{(void)advance(u,ledger,c);}
        catch(const Failure& e){
            rejected=true;
            require(e.stage.has_value()&&e.stage->stage=="interval",
                    "whole-step failure lost interval identity");
            for(const auto& stage : {"base","midpoint","endpoint"})
                require(has_array(e.partial_arrays,stage),
                        "whole-step failure lost stage state arrays");
            for(const auto& stage : {"base","midpoint","endpoint"})
                require(has_check(e.checks,"minimum_n",stage),
                        "whole-step failure lost stage diagnostics");
            require(has_check(e.checks,"CT_update","interval"),
                    "whole-step failure lost interval diagnostic");
            require_no_duplicate_checks(e.checks);
        }
        require(rejected,"negative control did not reject step");
        require(u.state.B==before.B&&u.state.fluid[0].n==before.fluid[0].n&&u.state.step==0&&ledger.accepted_intervals==0,"rejected step changed committed state");
    } else if(which=="midpoint_failure") {
        p.budgets.at("alias_perp.ion.linf").atol=-1;
        bool rejected=false;
        try{(void)advance(u,ledger,c);}
        catch(const Failure& e){
            rejected=true;
            require(e.stage.has_value()&&e.stage->stage=="midpoint",
                    "midpoint failure lost stage identity");
            require(has_array(e.partial_arrays,"base"),
                    "midpoint failure lost completed base state");
            require(has_array(e.partial_arrays,"midpoint"),
                    "midpoint failure lost trial state");
            require(has_check(e.checks,"minimum_n","endpoint",true),
                    "midpoint failure lost unavailable endpoint diagnostics");
            require_no_duplicate_checks(e.checks);
        }
        require(rejected,"midpoint negative control did not reject");
        require(u.state.B==before.B&&u.state.fluid[0].n==before.fluid[0].n&&
                u.state.step==0&&ledger.accepted_intervals==0,
                "midpoint rejection changed committed state");
    } else if(which=="source_failure") {
        auto source=[&](const State&,const StageTag& tag,Context& context) {
            auto rates=allocate_state(context);
            if(tag.stage=="midpoint")rates.B.clear();
            return rates;
        };
        bool rejected=false;
        try{(void)verification::advance(u,ledger,c,source);}
        catch(const Failure& e){
            rejected=true;
            require(e.code=="State","bad source failure code");
            require(e.stage.has_value()&&e.stage->stage=="midpoint",
                    "bad source failure lost midpoint identity");
            require(has_array(e.partial_arrays,"base_source"),
                    "bad source failure lost base source arrays");
            require(has_array(e.partial_arrays,"midpoint_source"),
                    "bad source failure lost malformed midpoint source arrays");
            require(e.partial_arrays.contains("midpoint_source_B")&&
                    e.partial_arrays.at("midpoint_source_B").empty(),
                    "bad source failure did not retain empty malformed B");
            require(has_check(e.checks,"minimum_n","midpoint"),
                    "bad source failure lost completed midpoint diagnostics");
            require_no_duplicate_checks(e.checks);
        }
        require(rejected,"bad source negative control did not reject");
        require(u.state.B==before.B&&u.state.fluid[0].n==before.fluid[0].n&&
                u.state.step==0&&ledger.accepted_intervals==0,
                "bad source rejection changed committed state");
    } else {
        for(std::uint64_t i=0;i<p.nsteps;++i) {
            auto result=advance(u,ledger,c);require(result.interval.accepted,"interval not committed");
            require(result.interval.right_step==i+1,"interval endpoint mismatch");
        }
        require(ledger.accepted_intervals==p.nsteps,"wrong number of accepted intervals");
        if(which=="equilibrium")require(max_abs(subtract(u.state.B,initial.B))<1e-15,"vacuum equilibrium moved");
        if(which=="evolution")require(norm(subtract(u.state.fluid[0].n,initial.fluid[0].n))>0,"nontrivial state did not evolve");
    }
    std::cout<<"PASS "<<which<<" solves="<<cpu.solve_calls<<" FFT="<<cpu.fft_calls<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
