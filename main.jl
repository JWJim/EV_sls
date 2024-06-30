using Timers,Random,Sobol
using Dates,Revise
# Random.seed!(Dates.today() |> Dates.value)
# using QuantEcon
# Pkg.add(["Polynomials","SpecialPolynomials","Distributions","Statistics","StatsBase"])
using DataFrames,CSV
using LinearAlgebra
using Polynomials,SpecialPolynomials
using Optim,Distributions,Statistics,StatsBase
using Base.Threads
using CompEcon
using Serialization
using KernelEstimator
# using Plots
## globals
include("gendata.jl") # data generating process
# random seed
# paras to be estimated
const beta  = 0.6                    # profit discount factor
# policy function approximation primitives
const max_poli_order = 3
# paras from other dataset
const d1 = 0.0; const d2 = 20.0; const d3 = 3.0;   # demand coefficient
function func_D(S_self,S_oppo) # charging demand
    return d1.+d2*(S_self[1]+S_oppo[1])+d3*(S_self[2]+S_oppo[2])
end

const num_of_alter = 500
const num_of_bootstrap = 200
  
function poly_transform(state_data) # transform using sieve
    function gen_poly(mat)
        retmat = hcat(ones(size(mat)[1]),mat)
        for i in 1:size(mat)[2]
            for j in i+1:size(mat)[2]
                retmat = hcat(retmat,mat[:,i].*mat[:,j]./10)
            end
        end
        return retmat
    end

    polinomial_eval = [basis(ChebyshevHermite,i).(state_data)./10^(i-1) for i in 1:max_poli_order]
    full_state = gen_poly(reduce(hcat, polinomial_eval))
    return full_state
end

function compress_state() # empirical distribution
    function check_S(eta,S)
        S_self_1 = S[:,1]
        S_self_2 = S[:,2]
        S_oppo_1 = S[:,3]
        S_oppo_2 = S[:,4]
        return func_D.(eachrow(hcat(S_self_1,S_self_2)),eachrow(hcat(S_oppo_1,S_oppo_2))).*(eta[1].+ eta[2].*S_self_1.+eta[3].*S_self_2.+eta[4].*S_oppo_1.+eta[5].*S_oppo_2)
    end

    function obj(eta,data)
        check_Sa = check_S(eta,hcat(data.sa1,data.sa2,data.sb1,data.sb2))
        check_Sb = check_S(eta,hcat(data.sb1,data.sb2,data.sa1,data.sa2))
        check_Sj = vcat(check_Sa,check_Sb)
        xj1 = vcat(data.xa1,data.xb1)
        xj2 = vcat(data.xa2,data.xb2)
        xj1_fit = npr(check_Sj, xj1, reg=locallinear, kernel=gaussiankernel)
        xj2_fit = npr(check_Sj, xj2, reg=locallinear, kernel=gaussiankernel)
        err1 = (xj1_fit.-xj1).^2 #.*(xj1.>0)
        err2 = (xj2_fit.-xj2).^2 #.*(xj2.>0)
        return mean(vcat(err1,err2))
    end

    data = CSV.read("generated_data.csv", DataFrame)
    eta_init = [20.0,1.0,1.0,-0.5,-1.0]
    eta_lower = [-20.0,-1.0,-1.0,-2.5,-3.0]
    eta_upper = [40.0,2.0,2.0,0.5,1.0]
    # res = optimize(eta->obj(eta,data),eta_lower,eta_upper,eta_init,Fminbox(LBFGS()),Optim.Options(x_tol=1e-8))
    res = optimize(eta->obj(eta,data),eta_init,LBFGS(),Optim.Options(x_tol=1e-8))
    eta_opt = res.minimizer
    

    # calculate results
    check_Sa = check_S(eta_opt,hcat(data.sa1,data.sa2,data.sb1,data.sb2))
    check_Sb = check_S(eta_opt,hcat(data.sb1,data.sb2,data.sa1,data.sa2))
    check_Sa_prime = check_S(eta_opt,hcat(data.sa1.+data.xa1,data.sa2.+data.xa2,data.sb1.+data.xb1,data.sb2.+data.xb2)) 
    check_Sb_prime = check_S(eta_opt,hcat(data.sb1.+data.xb1,data.sb2.+data.xb2,data.sa1.+data.xa1,data.sa2.+data.xa2)) 

    gmin = min(findmin(check_Sa)[1],findmin(check_Sb)[1])
    gmax = max(findmax(check_Sa)[1],findmax(check_Sb)[1])
    lin_space = LinRange(gmin,gmax,51)
    axis = diff(lin_space)/2+lin_space[1:end-1]
    return (data,check_Sa,check_Sb,check_Sa_prime,check_Sb_prime,axis,gmin,gmax,eta_opt) #,weight
end

function lom_reg(check_S_prime,check_S,x) # basic method for law of motion estimation
    x_poly = poly_transform(hcat(check_S,x))
    coeff = pinv(x_poly'*x_poly)*(x_poly'*check_S_prime) # OLS fit
    coeff = x_poly\check_S_prime
    return coeff
end

function lom_pred(coeff,check_S,x)
    x_poly = poly_transform(hcat(check_S,x))
    return x_poly*coeff
end

function tobit_reg(X,Y_mat)    # linear regression
    function log_llh_trans(coeff,X,y)
        rho   = coeff[size(X)[2]*2+1]
        sigma = exp(coeff[size(X)[2]*2+2])
        temp1 = (log.(y)-X*coeff[1:size(X)[2]])./sigma
        temp2 = X*coeff[size(X)[2]+1:size(X)[2]*2]
        if (abs(rho/sigma)>=1)
            return Inf
        else
            log_lhd = (logpdf.(Normal(),temp1).-log(sigma)+logcdf.(Normal(),(temp2+rho/sigma*temp1)/sqrt(1-(rho/sigma)^2))).*(y.>0.0)+logcdf.(Normal(),-temp2).*(y.<=0.0)
            return - mean(log_lhd)
        end
    end

    xx = poly_transform(X)
    coeff_mat = zeros((size(xx)[2]*2+2,size(Y_mat)[2]))
    final_llh = zeros(size(Y_mat)[2])
    for i in 1:size(Y_mat)[2]
        yy = Y_mat[:,i]
        c_init = vcat(xx[yy.>0,:]\log.(yy[yy.>0]),xx\((yy.>0).-0.5),[0.0,0.0])
        res = optimize(c->log_llh_trans(c,xx,yy),c_init,NelderMead(),Optim.Options(x_tol=1e-8))
        coeff_mat[:,i] = res.minimizer
        final_llh[i] = -res.minimum
    end
    return (coeff_mat,final_llh)
end

function tobit_pred(X,coeff_mat)  # linear prediction
    xx = poly_transform(X)
    temp1 = exp.(xx*coeff_mat[1:size(xx)[2],:])
    temp2 = xx*coeff_mat[size(xx)[2]+1:2*size(xx)[2],:]
    pred = temp1.*(temp2.>0)   
    return pred
end

function policy_approx(data,check_Sa,check_Sb;drop_index=-1)
    if drop_index>=1 # jackknife
        data_trim = copy(data)
        deleteat!(data_trim,drop_index)
        check_Sa_trim = copy(check_Sa)
        check_Sb_trim = copy(check_Sb)
        deleteat!(check_Sa_trim,drop_index)
        deleteat!(check_Sb_trim,drop_index)
        state_data_trim  = vcat(check_Sa_trim,check_Sb_trim)
        choice_data_trim = vcat(hcat(data_trim.xa1,data_trim.xa2,data_trim.xb1,data_trim.xb2),hcat(data_trim.xb1,data_trim.xb2,data_trim.xa1,data_trim.xa2))
        (pol_reg,final_llh) = tobit_reg(state_data_trim,choice_data_trim)
        return pol_reg
    else # first stage
        state_data  = vcat(check_Sa,check_Sb)
        choice_data = vcat(hcat(data.xa1,data.xa2,data.xb1,data.xb2),hcat(data.xb1,data.xb2,data.xa1,data.xa2))
        (pol_reg,final_llh) = tobit_reg(state_data,choice_data)
        randarray = randn(num_of_alter,size(pol_reg)...)
        pol_reg_alter = permutedims(repeat(pol_reg,outer=(1,1,num_of_alter)),(3,1,2))
        pol_reg_alter .+= randarray

        return (pol_reg,pol_reg_alter)
    end
end

function func_eval_order2_avg(paras,S,pol_reg,pol_reg_alter,lom_coeff,gmax,gmin)
    vc    = paras[1:2]
    ac    = paras[3:4]

    function func_cost_wac(a)
        return 0.5*vc[1]*((a[1]).^2).+0.5*vc[2]*((a[2]).^2)+ac[1]*(a[1].>0.0) + ac[2]*(a[2].>0.0)
    end
    function func_cost_noac(a)
        return 0.5*vc[1]*((a[1]).^2).+0.5*vc[2]*((a[2]).^2)
    end
            
    function func_eval_order2(nu,nu_next)
        # period 1
        invest_alter = tobit_pred(S,pol_reg_alter)
        invest_alter[invest_alter.>2*gmax] .= 2*gmax
        invest_real = tobit_pred(S,pol_reg)
        invest_real[invest_real.>2*gmax] .= 2*gmax
        a_p = invest_alter[:,1:2]
        Eb  = invest_real[:,3:end]
        b_p = invest_alter[:,3:end]
        Ea  = invest_real[:,1:2] 
        # period 2, a's deviation path
        S_next1 = lom_pred(lom_coeff,S,hcat(a_p,Eb))
        S_next1[S_next1.>gmax] .= gmax ### ????
        S_next1[S_next1.<gmin] .= gmin
        invest_alter_next1 = tobit_pred(S_next1,pol_reg_alter)
        invest_alter_next1[invest_alter_next1.>2*gmax] .= 2*gmax
        invest_real_next1 = tobit_pred(S_next1,pol_reg)
        invest_real_next1[invest_real_next1.>2*gmax] .= 2*gmax
        a_p_next = invest_alter_next1[:,1:2]
        Eb_next = invest_real_next1[:,3:end]
        S_next1_next = lom_pred(lom_coeff,S_next1,hcat(a_p_next,Eb_next))
        S_next1_next[S_next1_next.>gmax] .= gmax
        S_next1_next[S_next1_next.<gmax] .= gmin
        vala = -func_cost_wac.(eachrow(a_p)).+beta*S_next1.-beta*func_cost_noac.(eachrow(a_p_next)).+beta^2*S_next1_next*(1/(1-beta))
        # period 2, b's deviation path
        S_next2 = lom_pred(lom_coeff,S,hcat(b_p,Ea))
        S_next2[S_next2.>gmax] .= gmax
        S_next2[S_next2.<gmin] .= gmin
        invest_alter_next2 = tobit_pred(S_next2,pol_reg_alter)
        invest_alter_next2[invest_alter_next2.>2*gmax] .= 2*gmax
        invest_real_next2 = tobit_pred(S_next2,pol_reg)
        invest_real_next2[invest_real_next2.>2*gmax] .= 2*gmax
        b_p_next = invest_alter_next2[:,3:end]
        Ea_next  = invest_real_next2[:,1:2]
        S_next2_next = lom_pred(lom_coeff,S_next2,hcat(b_p_next,Ea_next))
        S_next2_next[S_next2_next.>gmax] .= gmax
        S_next2_next[S_next2_next.<gmin] .= gmin
        valb = -func_cost_wac.(eachrow(b_p)).+beta*S_next2.-beta*func_cost_noac.(eachrow(b_p_next)).+beta^2*S_next2_next*(1/(1-beta))
        return (vala,valb) 
    end
    
    (vala,valb) = func_eval_order2([0.0,0.0],[0.0,0.0])
    val = hcat(vala,valb)
    val[val.>9e9].= 9e9 # avoid too large results
    val[val.<-9e9].= -9e9 # avoid too small results
    return val'
end
    
function calc_T(paras,num_of_state,num_of_mkt,rndvec,S,gmin,gmax,lom_coeff,pol_reg,pol_reg_alter,pol_reg_jk;return_full=true)
    GC.gc()
    @show paras
    ## calculating Q_func
    val_0 = func_eval_order2_avg(paras,S,pol_reg,pol_reg,lom_coeff,gmax,gmin)
    val_alter = zeros(2,num_of_state,num_of_alter)
    mu_2 = zeros(2,num_of_state,num_of_alter)
    @threads for mi_k in 1:num_of_alter
        val_alter[:,:,mi_k] = func_eval_order2_avg(paras,S,pol_reg,pol_reg_alter[mi_k,:,:],lom_coeff,gmax,gmin)
    end 
    mu_2 = val_alter - repeat(val_0,outer=(1,1,num_of_alter))
    # mu_2 = calc_mean(mu_2,state_weight,num_of_pieces=2^4,dims=2) # average over all the state space with state_weight
    @show max_mu_2 = findmax(mu_2)[1]
    
    if return_full
        GC.gc()
        X_tilde = zeros(num_of_mkt,2,num_of_state)
        temp_jk = zeros(num_of_mkt,2,num_of_state)
        X_tilde_prime = zeros(num_of_mkt,2,num_of_state,num_of_alter)
        temp_alter_jk = zeros(num_of_mkt,2,num_of_state,num_of_alter)
        X_hat2 = zeros(num_of_mkt,2,num_of_state,num_of_alter)
        @threads for mi_i in 1:num_of_mkt
            temp_jk[mi_i,:,:] = func_eval_order2_avg(paras,S,pol_reg_jk[mi_i,:,:],pol_reg_jk[mi_i,:,:],lom_coeff,gmax,gmin)
        end
        ind_list = [(mi_i,mi_k) for mi_i in 1:num_of_mkt for mi_k in 1:num_of_alter]
        @threads for ind in ind_list
            temp_alter_jk[ind[1],:,:,ind[2]] = func_eval_order2_avg(paras,S,pol_reg_jk[ind[1],:,:],pol_reg_alter[ind[2],:,:],lom_coeff,gmax,gmin)
        end
        X_tilde = num_of_mkt*permutedims(repeat(val_0,outer=(1,1,num_of_mkt)),(3,1,2)).-(num_of_mkt-1)*temp_jk 
        X_tilde_prime = num_of_mkt*permutedims(repeat(val_alter,outer=(1,1,1,num_of_mkt)),(4,1,2,3)).-(num_of_mkt-1)*temp_alter_jk
        X_hat2 = X_tilde_prime - repeat(X_tilde,outer=(1,1,1,num_of_alter))
        @show findmax(X_hat2)[1] 

        # calculating variance func.
        sigma2_2 = var(X_hat2,mean=reshape(mu_2,(1,size(mu_2)...)),dims=1)
        sigma2_2[sigma2_2.<eps()] .= eps()
        norm_sim_temp = sqrt(num_of_mkt)*reshape(mu_2,(1,size(mu_2)...))./sqrt.(sigma2_2)
    
        GC.gc()
        T2 = findmax(norm_sim_temp)[1] 
        T = T2
        W = zeros(num_of_bootstrap)
        @threads for b in 1:num_of_bootstrap
            rndvec_expand = repeat(rndvec[b,:],outer=(1,size(X_hat2)[2:end]...))
            # temp2 = mean(rndvec_expand.*(X_hat2 .- mean(X_hat2,dims=1)),dims=1)
            temp2 = mean(rndvec_expand.*(X_hat2 .- reshape(mu_2,(1,size(mu_2)...))),dims=1)
            W[b] = findmax(sqrt(num_of_mkt)*temp2./sqrt.(sigma2_2))[1]
        end
        GC.gc()
        @show c95_direct = percentile(W,95) # 100-5+0.5*2 
        @show c_1step = percentile(W,99.5) # beta = 0.5 < 5/2
        # second step
        J_set_flag = norm_sim_temp .> -2*c_1step
        if sum(J_set_flag) == 0
            @show c95 = 0
        else
            @show sum(J_set_flag)/length(J_set_flag)
            W = zeros(num_of_bootstrap)
            GC.gc()
            @threads for b in 1:num_of_bootstrap
                rndvec_expand = repeat(rndvec[b+num_of_bootstrap,:],outer=(1,size(X_hat2)[2:end]...))
                # temp2 = rndvec_expand.*(X_hat2.-mean(X_hat2,dims=1))
                temp2 = rndvec_expand.*(X_hat2.-reshape(mu_2,(1,size(mu_2)...)))
                temp2 = temp2[repeat(J_set_flag,inner=(num_of_mkt,1,1,1))]
                temp2 = mean(reshape(temp2,(num_of_mkt,:)),dims=1)
                sigma_selected = sqrt.(sigma2_2)[J_set_flag]
                W[b] = findmax(sqrt(num_of_mkt)*temp2./sigma_selected')[1]
            end
            @show c95 = percentile(W,96) # 100-5+0.5*2 
        end
        open("mmstep.txt","a") do io
            println(io,paras," ",round(T;digits=8)," ",round(c95;digits=8))
        end
        println(paras," ",round(T;digits=8)," ",round(c95;digits=8))
        return (T,c95)
    else
        T = max_mu_2
        open("mmstep.txt","a") do io
            println(io,paras,"    ",round(T;digits=8))
        end
        println(paras," ",round(T;digits=8))
        return T
    end
end

function estimate_bbl()  # main procedure of estimation following CCK(2019)
    dt_temp = CSV.read("generated_data.csv", DataFrame)
    paras_true = [2000.0,1000.0,100.0,50.0]
    paras_lower = paras_true .* eps()
    paras_upper = paras_true .* 2.0
    GC.gc()

    (raw_data,check_Sa,check_Sb,check_Sa_prime,check_Sb_prime,bas_x,gmin,gmax,eta) = compress_state() # ,state_weight
    num_of_mkt = size(check_Sa)[1]
    (pol_reg,pol_reg_alter) = policy_approx(raw_data,check_Sa,check_Sb)
        
     # conduct jack-knife
    pol_reg_jk = zeros(num_of_mkt,size(pol_reg)...)
    @threads for i in 1:num_of_mkt
        pol_reg_jk[i,:,:] = policy_approx(raw_data,check_Sa,check_Sb;drop_index=i)
    end
    
    # calculate law of motion
    state_data_check  = vcat(check_Sa,check_Sb)
    choice_data = vcat(hcat(raw_data.xa1,raw_data.xa2,raw_data.xb1,raw_data.xb2),hcat(raw_data.xb1,raw_data.xb2,raw_data.xa1,raw_data.xa2))
    state_data_prime_check = vcat(check_Sa_prime,check_Sb_prime)
    lom_coeff = lom_reg(state_data_prime_check,state_data_check,choice_data)
    
    all_state = bas_x
    num_of_state = size(all_state)[1]

    rndvec = randn(num_of_bootstrap*2,num_of_mkt)
    calc_T(paras_true,rndvec,return_full=true)
    for i in 0.1:0.1:2.0
        GC.gc()
        paras_test = copy(paras_true) * i
        paras_test[5:7] = copy(paras_true)[5:7]
        calc_T(paras_test,num_of_state,num_of_mkt,rndvec,all_state,gmin,gmax,lom_coeff,pol_reg,pol_reg_alter,pol_reg_jk,return_full=true)
    end
    return 0
end

function main()
    # for i in 1:100
        GC.gc()
        gendata(Int(rand(1:1e8)),num_of_obs = 1000)
        estimate_bbl()
        GC.gc()
    # end
end

main()