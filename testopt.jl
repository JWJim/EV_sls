using Timers,Random,Sobol
using Dates
using Revise
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
using Plots
## globals
include("gendata.jl") # data generating process
# random seed
# paras to be estimated
const beta  = 0.6                    # profit discount factor
# policy function approximation primitives
const max_poli_order = 2
# paras from other dataset
const d1 = 0.0; const d2 = 20.0; const d3 = 3.0;   # demand coefficient
function func_D(S_self,S_oppo) # charging demand
    return d1.+d2*(S_self[1]+S_oppo[1])+d3*(S_self[2]+S_oppo[2])
end

const num_of_alter = 500
const num_of_bootstrap = 200

# function poly_transform(state_data) # transform using sieve
#     function gen_poly(mat)
#         retmat = hcat(ones(size(mat)[1]),mat./1)
#         for i in 1:size(mat)[2]
#             for j in i:size(mat)[2]
#                 retmat = hcat(retmat,mat[:,i].*mat[:,j]./10)
#             end
#         end
#         return retmat
#     end

#     polinomial_eval = [basis(ChebyshevHermite,i).(state_data)./10^(i-1) for i in 1:max_poli_order]
#     full_state = gen_poly(reduce(hcat, polinomial_eval))
#     return full_state
# end

function compress_state(eta) # empirical distribution
    function check_S(S_self_1,S_self_2,S_oppo_1,S_oppo_2)
        return eta[1].+ eta[2].*S_self_1 .+ eta[3].*S_self_2 .+ eta[4].*S_oppo_1 .+ eta[5].*S_oppo_2
    end

    data = CSV.read("generated_data.csv", DataFrame)
    check_Sa = reduce(vcat,check_S.(eachrow(data.sa1),eachrow(data.sa2),eachrow(data.sb1),eachrow(data.sb2)))
    check_Sb = reduce(vcat,check_S.(eachrow(data.sb1),eachrow(data.sb2),eachrow(data.sa1),eachrow(data.sa2)))
    check_Sa_prime = reduce(vcat,check_S.(eachrow(data.sa1.+data.xa1),eachrow(data.sa2.+data.xa2),eachrow(data.sb1.+data.xb1),eachrow(data.sb2.+data.xb2)))
    check_Sb_prime = reduce(vcat,check_S.(eachrow(data.sb1.+data.xb1),eachrow(data.sb2.+data.xb2),eachrow(data.sa1.+data.xa1),eachrow(data.sa2.+data.xa2)))

    gmin = min(findmin(check_Sa)[1],findmin(check_Sb)[1])
    gmax = max(findmax(check_Sa)[1],findmax(check_Sb)[1])
    lin_space = LinRange(gmin,gmax,41)
    axis = diff(lin_space)/2+lin_space[1:end-1]
    return (data,check_Sa,check_Sb,check_Sa_prime,check_Sb_prime,axis) #,weight
end

function lom_reg(check_S_prime,check_S,x) # basic method for law of motion estimation
    coeff = hcat(ones(size(check_S)[1]),check_S,x)\check_S_prime
    return coeff
end

function lom_pred(coeff,check_S,x)
    return hcat(ones(size(check_S)[1]),check_S,x)*coeff
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
    coeff_mat = zeros((2*2+2,size(Y_mat)[2]))
    final_llh = zeros(size(Y_mat)[2])
    for i in 1:size(Y_mat)[2]
        yy = Y_mat[:,i]
        xx = hcat(ones(size(X)[1]),X)
        c_init = vcat(xx[yy.>0,:]\log.(yy[yy.>0]),xx\((yy.>0).-0.5),[0.0,0.0])
        res = optimize(c->log_llh_trans(c,xx,yy),c_init,NelderMead(),Optim.Options(x_tol=1e-8))
        coeff_mat[:,i] = res.minimizer
        final_llh[i] = -res.minimum
    end
    return (coeff_mat,final_llh)
end

function tobit_pred(X,coeff_mat)  # linear prediction
    xx = hcat(ones(size(X)[1]),X)
    temp1 = exp.(xx*coeff_mat[1:2,:])
    temp2 = xx*coeff_mat[3:4,:]
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
# def func_eval_order1_avg(paras,S,pol_reg,pol_reg_alter,mi_1,mi_2=-1):
#     vc    = paras[:2]
#     fc    = paras[2:4]
#     gamma = paras[4:]
    
#     def func_T_share(S_self,S_oppo,gamma):
#         return S_self/(S_self + S_oppo + gamma)
#     def func_T_reduced(S_self,S_oppo,locisrural):
#         if locisrural == 0:
#             re = 1.0
#         else:
#             re = 1.0
#         return re
#     def func_revenue(S_self,S_oppo): # revenue function
#         return  \
#             + func_D(S_self,S_oppo)*func_T_reduced(S_self,S_oppo,0)*func_T_share(S_self[0],S_oppo[0],gamma[0]) \
#             + func_D(S_self,S_oppo)*func_T_reduced(S_self,S_oppo,1)*func_T_share(S_self[1],S_oppo[1],gamma[1])
#     def func_cost(nu,a,S_self):
#         return 0.5*np.exp(nu[0])*vc[0]*((a[0])**2)+0.5*np.exp(nu[1])*vc[1]*((a[1])**2) \
#             + fc[0]*(a[0]>0.0) + fc[1]*(a[1]>0.0)
            
#     def func_eval_order1(nu):
#         Sa = S[:2]
#         Sb = S[2:]
#         x_poly = poly_transform(S.T)
#         temp = np.maximum(tobit_pred(x_poly,pol_reg_alter).T,gmin)
#         invest_alter = np.minimum(temp,gmax)
#         temp = np.maximum(tobit_pred(x_poly,pol_reg).T,gmin)
#         invest_real = np.minimum(temp,gmax)
#         a_p = invest_alter[:2]
#         Eb  = invest_real[2:]
#         vala = - func_cost(nu,a_p,Sa) + beta*func_revenue(Sa+a_p,Sb+Eb)
#         b_p = invest_alter[2:]
#         Ea  = invest_real[:2]
#         valb = - func_cost(nu,b_p,Sb) + beta*func_revenue(Sb+b_p,Sa+Ea)
#         return (vala,valb) 
    
#     val = np.empty((gqn**2,2,num_of_state))
#     for i in range(gqn**2):
#         (vala,valb) = func_eval_order1(nu[:,i])
#         val[i,:,:] = np.vstack((vala,valb))
#     val = np.average(val,axis=0,weights=we)
#     if mi_2==-1:
#         return (mi_1,val)
#     else:
#         return (mi_1,mi_2,val)

function func_eval_order2_avg(paras,S,pol_reg,pol_reg_alter,lom_coeff)
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
        invest_real = tobit_pred(S,pol_reg)
        a_p = invest_alter[:,1:2]
        Eb  = invest_real[:,3:end]
        b_p = invest_alter[:,3:end]
        Ea  = invest_real[:,1:2] 
        # period 2, a's deviation path
        S_next1 = lom_pred(lom_coeff,S,hcat(a_p,Eb))
        S_next1[S_next1.>gmax] .= gmax ### ????
        S_next1[S_next1.<gmin] .= gmin
        invest_alter_next1 = tobit_pred(S_next1,pol_reg_alter)
        invest_real_next1 = tobit_pred(S_next1,pol_reg)
        a_p_next = invest_alter_next1[:,1:2]
        Eb_next  = invest_real_next1[:,3:end]
        S_next1_next = lom_pred(lom_coeff,S_next1,hcat(a_p_next,Eb_next))
        S_next1_next[S_next1_next.>gmax] .= gmax
        S_next1_next[S_next1_next.<gmax] .= gmin
        vala = -func_cost_wac.(eachrow(a_p)).+beta*S_next1.-beta*func_cost_noac.(eachrow(a_p_next)).+beta^2*S_next1_next*(1/(1-beta))
        # period 2, b's deviation path
        S_next2 = lom_pred(lom_coeff,S,hcat(b_p,Ea))
        S_next2[S_next2.>gmax] .= gmax
        invest_alter_next2 = tobit_pred(S_next2,pol_reg_alter)
        invest_real_next2 = tobit_pred(S_next2,pol_reg)
        b_p_next = invest_alter_next2[:,3:end]
        Ea_next  = invest_real_next2[:,1:2]
        S_next2_next = lom_pred(lom_coeff,S_next2,hcat(b_p_next,Ea_next))
        S_next2_next[S_next2_next.>gmax] .= gmax
        S_next2_next[S_next2_next.<gmin] .= gmin
        valb = -func_cost_wac.(eachrow(b_p)).+beta*S_next2.-beta*func_cost_noac.(eachrow(b_p_next)).+beta^2*S_next2_next*(1/(1-beta))
        return (vala,valb) 
    end
    
    # val = np.empty((gqn**2,2,num_of_state))
#   for i in range(gqn**2):
#       for j in range(gqn**2):
#       (vala,valb) = func_eval_order2(nu[:,i],nu[:,j])
#       val = val + np.vstack((vala,valb))*we[i]*we[j]
    # for i in range(gqn**2):
    #     (vala,valb) = func_eval_order2(nu[:,i],nu[:,i])
    #     val[i,:,:] = np.vstack((vala,valb))
    # val = np.average(val,axis=0,weights=we)
    (vala,valb) = func_eval_order2([0.0,0.0],[0.0,0.0])
    val = hcat(vala,valb)
    return val'
end
    
function calc_T(paras,rndvec;return_full=true)
    GC.gc()
    @show eta = paras[5:end]
    
    (raw_data,check_Sa,check_Sb,check_Sa_prime,check_Sb_prime,bas_x) = compress_state(eta) # ,state_weight
    num_of_mkt = size(check_Sa)[1]
    (pol_reg,pol_reg_alter) = policy_approx(raw_data,check_Sa,check_Sb)
        
     # conduct jack-knife
    pol_reg_jk = zeros(num_of_mkt,size(pol_reg)...)
    @threads for i in 1:num_of_mkt
        pol_reg_jk[i,:,:] = policy_approx(raw_data,check_Sa,check_Sb;drop_index=i)
    end
    
    state_data_check  = vcat(check_Sa,check_Sb)
    choice_data = vcat(hcat(data.xa1,data.xa2,data.xb1,data.xb2),hcat(data.xb1,data.xb2,data.xa1,data.xa2))
    state_data_prime_check = vcat(check_Sa_prime,check_Sb_prime)
    lom_coeff = lom_reg(state_data_prime_check,state_data_check,choice_data)
    
    all_state = bas_x; S = bas_x
    num_of_state = size(all_state)[1]
        
    # function calc_mean(arr,wt;num_of_pieces=1,dims=1)
    #     cl = Int(size(arr)[dims]/num_of_pieces)
    #     re = [mean(selectdim(arr,dims,(i-1)*cl+1:i*cl),weights(wt[(i-1)*cl+1:i*cl]),dims=dims) for i in 1:num_of_pieces]
    #     return cat(re...,dims=dims)
    # end
    
    ## calculating Q_func
    val_0 = func_eval_order2_avg(paras,S,pol_reg,pol_reg,lom_coeff)
    val_alter = zeros(2,num_of_state,num_of_alter)
    mu_2 = zeros(2,num_of_state,num_of_alter)
    @threads for mi_k in 1:num_of_alter
        val_alter[:,:,mi_k] = func_eval_order2_avg(paras,S,pol_reg,pol_reg_alter[mi_k,:,:],lom_coeff)
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
            temp_jk[mi_i,:,:] = func_eval_order2_avg(paras,S,pol_reg_jk[mi_i,:,:],pol_reg_jk[mi_i,:,:],lom_coeff)
        end
        ind_list = [(mi_i,mi_k) for mi_i in 1:num_of_mkt for mi_k in 1:num_of_alter]
        @threads for ind in ind_list
            temp_alter_jk[ind[1],:,:,ind[2]] = func_eval_order2_avg(paras,S,pol_reg_jk[ind[1],:,:],pol_reg_alter[ind[2],:,:],lom_coeff)
        end
        X_tilde = num_of_mkt*permutedims(repeat(val_0,outer=(1,1,num_of_mkt)),(3,1,2)).-(num_of_mkt-1)*temp_jk 
        X_tilde_prime = num_of_mkt*permutedims(repeat(val_alter,outer=(1,1,1,num_of_mkt)),(4,1,2,3)).-(num_of_mkt-1)*temp_alter_jk
        X_hat2 = X_tilde_prime - repeat(X_tilde,outer=(1,1,1,num_of_alter))
        @show findmax(X_hat2)[1] 
        # X_hat2 = calc_mean(X_hat2,state_weight,num_of_pieces=2^4,dims=3) # average over all the state space with state_weight

        # calculating variance func.
        sigma2_2 = var(X_hat2,mean=reshape(mu_2,(1,size(mu_2)...)),dims=1)
        # sigma2_2 = var(X_hat2,dims=1)
        sigma2_2[sigma2_2.<eps()] .= eps()
        norm_sim_temp = sqrt(num_of_mkt)*reshape(mu_2,(1,size(mu_2)...))./sqrt.(sigma2_2)
    
        plt = histogram(reshape(norm_sim_temp,:), bins=range(-5,5,length=21), normalize=:pdf, color=:gray)
        savefig(plt,"para_1_"*string(paras[1])*".png")
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

# function save_temp_data(num_of_mkt,num_of_state,rndvec,all_state,state_weight,pol_reg,pol_reg_alter,pol_reg_jk)
#     if !isdir("temp_data")
#         mkdir("temp_data")
#     end
#     serialize("temp_data/"*"num_of_mkt.dat",num_of_mkt)
#     serialize("temp_data/"*"num_of_state.dat", num_of_state)
#     serialize("temp_data/"*"all_state.dat", all_state)
#     serialize("temp_data/"*"rndvec.dat", rndvec)
#     serialize("temp_data/"*"state_weight.dat", state_weight)
#     serialize("temp_data/"*"pol_reg.dat", pol_reg)
#     serialize("temp_data/"*"pol_reg_alter.dat", pol_reg_alter)
#     serialize("temp_data/"*"pol_reg_jk.dat", pol_reg_jk)
#     return 0
# end

# function read_temp_data()
#     num_of_mkt    = deserialize("temp_data/"*"num_of_mkt.dat")
#     num_of_state  = deserialize("temp_data/"*"num_of_state.dat")
#     all_state     = deserialize("temp_data/"*"all_state.dat")
#     rndvec        = deserialize("temp_data/"*"rndvec.dat")
#     state_weight  = deserialize("temp_data/"*"state_weight.dat")
#     pol_reg       = deserialize("temp_data/"*"pol_reg.dat")
#     pol_reg_alter = deserialize("temp_data/"*"pol_reg_alter.dat")
#     pol_reg_jk    = deserialize("temp_data/"*"pol_reg_jk.dat")
#     return(num_of_mkt,num_of_state,rndvec,all_state,state_weight,pol_reg,pol_reg_alter,pol_reg_jk)
# end

function estimate_bbl()  # main procedure of estimation following CCK(2019)
## first stage estimation of policy functions
    # read data
    # open("mmstep.txt","w") do io
    #     println(io,"Start!")
    # end

    # drop those parameters covering 0 
    # pol_reg_min = zeros(size(pol_reg))
    # pol_reg_max = zeros(size(pol_reg))
    # for i in 1:size(pol_reg_jk)[2]
    #     for j in 1:size(pol_reg_jk)[3]
    #         pol_reg_min[i,j] = percentile(pol_reg_jk[:,i,j],2.5)
    #         pol_reg_max[i,j] = percentile(pol_reg_jk[:,i,j],97.5)
    #     end
    # end
    # flag = (pol_reg_min.*pol_reg_max.>0)
    # pol_reg = pol_reg.*flag
    # pol_reg_alter = pol_reg_alter.*permutedims(repeat(flag,outer=(1,1,num_of_alter)),(3,1,2))
    # pol_reg_jk = pol_reg_jk.*permutedims(repeat(flag,outer=(1,1,num_of_mkt)),(3,1,2))

    # conduct multiplier bootstrap
    rndvec = randn(num_of_bootstrap*2,num_of_mkt)
    paras_true = [2000.0,1000.0,100.0,50.0,0.5,1.0]
    paras_lower = paras_true .* eps()
    paras_upper = paras_true .* 2.0
#     all_state = vcat(bas_x,hcat(data_raw.sa1,data_raw.sa2,data_raw.sb1,data_raw.sb2))
    # all_state = hcat(data_raw.sa1,data_raw.sa2,data_raw.sb1,data_raw.sb2)
    GC.gc()
    # save_temp_data(num_of_mkt,num_of_state,rndvec,all_state,state_weight,pol_reg,pol_reg_alter,pol_reg_jk)
#     res = optimize(x->calc_T(x,num_of_mkt,rndvec,bas_x,state_weight,pol_reg,pol_reg_alter,pol_reg_jk,return_full=false),paras_lower,paras_upper,paras_true,SAMIN(rt=0.5),Optim.Options(iterations=10^6))
    calc_T(paras_true*1.0,rndvec,return_full=true)
    GC.gc()
    calc_T(paras_true*0.1,rndvec,return_full=true)
    GC.gc()
    calc_T(paras_true*2.0,rndvec,return_full=true)
    GC.gc()
    calc_T(paras_true*1e-5,rndvec,return_full=true)
    GC.gc()
    calc_T(paras_true*1e5,rndvec,return_full=true)
    GC.gc()
    # res = optimize(x->calc_T(x,num_of_mkt,num_of_state,rndvec,all_state,state_weight,pol_reg,pol_reg_alter,pol_reg_jk,return_full=false),paras_lower,paras_upper,paras_true,Fminbox(LBFGS()),Optim.Options(x_tol=1e-4))
    # paras_opt = res.minimizer
    # if Optim.converged(res) == false
    #     println("Not success")
    #     println(res)
    # end
    # calc_T(paras_opt,num_of_mkt,num_of_state,rndvec,all_state,state_weight,pol_reg,pol_reg_alter,pol_reg_jk,return_full=true)
    # open("mmstep.txt","a") do io
    #     println(io,"res = ",round.(paras_opt;digits=4))
    # end
    # serialize("temp_data/"*"opt_results.dat",paras_opt)
    # serialize("temp_data/"*"paras_lower.dat",paras_lower)
    # serialize("temp_data/"*"paras_upper.dat",paras_upper)
    return 0
end

function main()
    # if !isfile("generated_data.csv")
    #     gendata(Int(rand(1:1e8)),num_of_obs = 200)
    # end
    # estimate_bbl()
    for i in 1:1
        GC.gc()
        gendata(Int(rand(1:1e8)),num_of_obs = 200)
        estimate_bbl()
        GC.gc()
    end
end

main()