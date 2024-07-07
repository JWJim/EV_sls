# all reauired packages
# using Pkg
# Pkg.add(["Random","DataFrames","CSV","LinearAlgebra","CompEcon","Optim","Timers"])
# randomseed
# using Timers
# using Random
# # Random.seed!(20240428)
# # use packages
# using DataFrames,CSV
# using LinearAlgebra
# using CompEcon
# using Optim

function gendata(RNG_seed;num_of_obs=750)
    Random.seed!(RNG_seed)
    # Globals
    ## globals
    # paras to be estimated
    vc    = [200.0,100.0];              # quadratic investment cost
    ac    = [10.0,5.0];                   # fixed cost
    beta  = 0.6;                      # profit discount factor
    gamma = [0.025,0.05];              # reduced form
    # paras from other dataset
    d1 = 0.0; d2 = 20.0; d3 = 3.0;   # demand coefficient
    
    # func approx. grids
    gmin = 0.0;
    gmax = 10.0;
    nn,aa,bb = [6,6,6,6], [gmin,gmin,gmin,gmin], [gmax,gmax,gmax,gmax];
    basis = fundefn(:spli, nn, aa, bb)
    bas_x = funnode(basis)[1]
    phi = funbase(basis)
    
    function func_D(S_self,S_oppo) # charging demand
        return d2*10.0+d3*10.0
    end
    
    function func_revenue(S_self,S_oppo) # revenue function
        return func_D(S_self,S_oppo).*(S_self[1].*(1.0.-gamma[1]*S_oppo[1]).+S_self[2].*(1.0.-gamma[2]*S_oppo[2]))
    end
    function func_cost_wac(a)
        return 0.5*vc[1]*((a[1]).^2).+0.5*vc[2]*((a[2]).^2)+ac[1]*(a[1].>0.0) + ac[2]*(a[2].>0.0)
    end
    function func_cost_noac(a)
        return 0.5*vc[1]*((a[1]).^2).+0.5*vc[2]*((a[2]).^2)
    end
        
    function func_V3(;S3 = bas_x) # value function of period 3
        temp = func_revenue.(eachrow(S3[:,1:2]),eachrow(S3[:,3:end]))*(1/(1-beta))
        return temp
    end
    
    function func_V(;S = bas_x,V_next,simu)
        function func_eval(invest,S_i,Eb_i)
            ES_next = S_i + vcat(invest,Eb_i)
            ES_next[ES_next.>gmax] .= gmax
            temp = funeval(V_next,basis,ES_next)[1]
            val = func_revenue(S_i[1:2],S_i[3:end]).- func_cost_wac(invest).+beta*temp
            return -val[1] # notice that now we do not need to take the inverses
        end
        
        function func_eval_noac(invest,S_i,Eb_i)
            ES_next = S_i + vcat(invest,Eb_i)
            ES_next[ES_next.>gmax] .= gmax
            temp = funeval(V_next,basis,ES_next)[1]
            val = func_revenue(S_i[1:2],S_i[3:end]).- func_cost_noac(invest).+beta*temp
            return -val[1]
        end
                
        function func_V_opt(S_i,Ea_i,Eb_i)
            lower = [gmin,gmin]
            upper = [gmax,gmax]
            res = optimize(x->func_eval_noac(x,S_i,Eb_i),lower,upper,Ea_i,Fminbox(LBFGS()),Optim.Options(x_tol=1e-8,iterations=999999))
            if Optim.converged(res) == false
                println("Solve model Not success")
                println(res)
            end
            invest = res.minimizer
            return invest
        end
            
        function func_V_withbd(S_i,Ea_i,Eb_i)
            # lower = [gmin,gmin]
            # upper = [gmax,gmax]
            # res = optimize(x->func_eval_noac(x,S_i,Eb_i),lower,upper,Ea_i,Fminbox(LBFGS()))
            # if Optim.converged(res) == false
            #     println("Not success")
            #     println(res)
            # end
            invest = Ea_i
            invest_11 = invest
            val_11 = -func_eval(invest,S_i,Eb_i)
            invest_01 = zeros(Int(size(S_i)[1]/2))
            invest_01[2] = invest[2]
            val_01 = -func_eval(invest_01,S_i,Eb_i)
            invest_10 = zeros(Int(size(S_i)[1]/2))
            invest_10[1] = invest[1]
            val_10 = -func_eval(invest_10,S_i,Eb_i)
            val_mat = [val_11-val_01,val_11-val_10]
            invest = (val_mat.>0.0).*invest_11
            return invest
        end
        
        Ea = ones(size(S)[1],Int(size(S)[2]/2))*(gmin+gmax)*0.5
        Eb = ones(size(S)[1],Int(size(S)[2]/2))*(gmin+gmax)*0.5
        S_tilde = hcat(S[:,3:end],S[:,1:2])
        
        function find_fixedpoint!(Ea_i,Eb_i,S_i,S_tilde_i)
            Ea_tmp = fill(eps(),size(Ea_i)); Eb_tmp = fill(eps(),size(Ea_i))
            iter = 0
            while (norm((Ea_tmp-Ea_i)./Ea_i)>1e-6) && (norm((Eb_tmp-Eb_i)./Eb_i)>1e-6) && (iter<1e6)
                iter += 1
                Ea_tmp = Ea_i;Eb_tmp=Eb_i;
                optimizer = func_V_opt(S_i,Ea_i,Eb_i)
                Ea_i = 0.5*Ea_i .+ 0.5*optimizer
                optimizer = func_V_opt(S_tilde_i,Eb_i,Ea_i)
                Eb_i = 0.5*Eb_i .+ 0.5*optimizer
            end
            return (Ea_i,Eb_i)
        end
    
        Threads.@threads for i in 1:size(S)[1]
            (Ea[i,:],Eb[i,:]) = find_fixedpoint!(Ea[i,:],Eb[i,:],S[i,:],S_tilde[i,:])
        end
        
        if simu == false # that means it is calculating on the grids
            V = -func_eval_noac.(eachrow(Ea),eachrow(S),eachrow(Eb))
            return V
        else # that means we are now simulating, N.B. only in the final desision, fc shows up
            real_a = zeros(size(Ea))
            real_b = zeros(size(Eb))
            Threads.@threads for i in 1:size(S)[1]
                real_a[i,:] = func_V_withbd(S[i,:],Ea[i,:],Eb[i,:])
                real_b[i,:] = func_V_withbd(S_tilde[i,:],Eb[i,:],Ea[i,:])
            end
            return (real_a,real_b)
        end
    end
    
    function generate_func_approx_last()
        # building the functional approximation 
        y = func_V3()
        c = phi\y
        return c
    end
    
    function generate_func_approx(c_next;simu = false) 
        y = func_V(V_next = c_next, simu = simu)
        c = phi\y
        return c
    end
    
    function simulate_data(num_of_obs,c_next)
        S1 = rand(num_of_obs,4)*(gmax-2.0)
        # sobol_seq = SobolSeq(4);skip(sobol_seq,num_of_obs+RNG_seed)
        # S1= reduce(hcat, [next!(sobol_seq)*(gmax-2.0) for i in 1:num_of_obs])'
        (a1,b1) = func_V(S=S1, V_next = c_next, simu = true)
        return hcat(S1,a1,b1)
    end
    
    function write_data(generated_data)
        df = DataFrame(generated_data,["sa1","sa2","sb1","sb2","xa1","xa2","xb1","xb2"])
        CSV.write("generated_data.csv",df)
    end

    # start main
    tic();
    v3_approx = generate_func_approx_last()
    @time v2_approx = generate_func_approx(v3_approx)
    @time generated_data = simulate_data(num_of_obs,v2_approx)
    write_data(generated_data)
    open("log.txt","a") do io
        println(io,"Time used:",toc())
        println(io,"Num of threads",Threads.nthreads())
    end
end

# main()
