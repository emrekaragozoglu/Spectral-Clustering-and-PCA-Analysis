%%
%QUESTION 1
%PART A

load hw4_data1

sum_rows=sum(M,1);
mean_rows=sum_rows/100;

for i=1:1:100
    centered_M(i,:)=M(i,:)-mean_rows;
end


[U,S,svd_M] = svd(centered_M);
pca_M = pca(centered_M);

for i=1:1:100
    [Vec,Diag]=eig((centered_M(i,:))'*centered_M(i,:));
    %eigenvalues_vectors(i)=max(E);
end

eigenvalues_cov_matrix=eig(transpose(centered_M)*centered_M);
k=1:4;
figure(1)
plot(k,eigenvalues_cov_matrix);
title('Eigen Values as a function of k');
xlabel('k')
ylabel('Eigen Value')

eigenvalues_cov_matrix
%%
%PART C
[V,D] = eig(transpose(centered_M)*centered_M);
max_eigenvector=V(:,4);

for i=1:1:100
    projections(i)=dot(max_eigenvector,centered_M(i,:));
end

figure(2)
histogram(projections,100)
title('Histogram of the inner product of the data vectors with the eigenvector corresponding to the maximal eigenvalue')
summ=0;
for i=1:1:100
    summ=summ+(projections(i)^2);
end
sumofsquares=summ
disp('D(4,4)')
D(4,4)
disp('The sum of squares of dot product values are equivalent to lambda 1.')

%%
%PART D
max_eigenvector2=V(:,3);
for i=1:1:100
    projections2(i)=dot(max_eigenvector2,centered_M(i,:));
end

figure(3)
scatter(projections,projections2);
title('Scatter plot of recentered data projected onto the first two principal components')
axis equal
xlabel('')
ylabel('')

summ2=0;
for i=1:1:100
    summ2=summ2+(projections2(i)^2);
end
sumofsquares2=summ2;

disp('sumofsquares+sumofsquares2')
sumofsquares+sumofsquares2

disp('D(4,4)+D(3,3)');
D(4,4)+D(3,3)
disp('The sum of squared lengths of dot product values are equivalent to lambda1+lambda2.')

%%
%PART E

max_eigenvector3=V(:,2);
max_eigenvector4=V(:,1);

for i=1:1:100
    projections3(i)=dot(max_eigenvector3,centered_M(i,:));
    projections4(i)=dot(max_eigenvector4,centered_M(i,:));
end

summ3=0;
summ4=0;
summ1=0;
summ2=0;

for i=1:1:100
    summ1=summ1+(projections(i)^2);
    summ2=summ2+(projections2(i)^2);
    summ3=summ3+(projections3(i)^2);
    summ4=summ4+(projections4(i)^2);
end

summ_array(1)=summ1/(summ1+summ2+summ3+summ4);
summ_array(2)=summ2/(summ1+summ2+summ3+summ4);
summ_array(3)=summ3/(summ1+summ2+summ3+summ4);
summ_array(4)=summ4/(summ1+summ2+summ3+summ4);

figure(4)
k=1:1:4;
bar(k,summ_array)
title('Proportion of variance as a function of component order')
xlabel('Component Order')
ylabel('Proportion of variance')
%%
%QUESTION 2
%PART A
load hw4_data1

for iteration=1:1:500
    for c=1:1:4
        random_indices=randperm(100);
        for i=1:1:100
            permuted_M(i,c)=M(random_indices(i),c);
        end
    end
    
    [eig_vecs,eig_vals]=pca(permuted_M);
    
    for col=1:1:4
        var_vec(col)=var(eig_vals(:,col));
    end
    var_vec=var_vec./(sum(var_vec));
    
    var_matrix(:,iteration)=var_vec;
end

for l=1:1:4
    total_var_vec(l)=mean(var_matrix(l,:));
end

[eig_vecs_M,eig_vals_M]=pca(M);

for col=1:1:4
    var_vec_M(col)=var(eig_vals_M(:,col));
end
total_var_vec_M=var_vec_M./(sum(var_vec_M));


figure(1)    
bar(total_var_vec_M,0.5,'r')
hold on
bar(total_var_vec,0.25);
legend('Proportion of variance for M','Means of the proportion of variance for permuted M')
title('Bar Plot of the proportion of variance')
xlabel('Stimulus conditions')

%%
%PART B

for iteration=1:1:500
    random_column=ceil(4*rand());
    random_indices=randperm(100);
    permuted_M=M;
    
    for i=1:1:100
        permuted_M(i,random_column)=M(random_indices(i),random_column);
    end

    [eig_vecs,eig_vals]=pca(permuted_M);
    
    for col=1:1:4
        var_vec(col)=var(eig_vals(:,col));
    end
    var_vec=var_vec./(sum(var_vec));
    
    var_matrix(:,iteration)=var_vec;
end

for l=1:1:4
    total_var_vec(l)=mean(var_matrix(l,:));
end

[eig_vecs_M,eig_vals_M]=pca(M);

for col=1:1:4
    var_vec_M(col)=var(eig_vals_M(:,col));
end
total_var_vec_M=var_vec_M./(sum(var_vec_M));

figure(2)    
bar(total_var_vec_M,0.5,'r')
hold on
bar(total_var_vec,0.25);
legend('Proportion of variance for M','Means of the proportion of variance for permuted M')
title('Bar Plot of the proportion of variance')
xlabel('Stimulus conditions')

%%
%PART C
for column=1:1:4
    for iteration=1:1:500
        random_indices=randperm(100);
        permuted_M=M;
        for i=1:1:100
            permuted_M(i,column)=M(random_indices(i),column);
        end
        
        [eig_vecs,eig_vals]=pca(permuted_M);
        
        for col=1:1:4
            var_vec(col)=var(eig_vals(:,col));
        end
        var_vec=var_vec./(sum(var_vec));
        
        var_matrix(:,iteration)=var_vec;
    end
    
    for l=1:1:4
        total_var_vec(l,column)=mean(var_matrix(l,:));
    end
end

[eig_vecs_M,eig_vals_M]=pca(M);

for col=1:1:4
    var_vec_M(col)=var(eig_vals_M(:,col));
end
total_var_vec_M=var_vec_M./(sum(var_vec_M));

figure(3)    
subplot(2,2,1)
bar(total_var_vec_M,0.5,'r')
hold on
bar(total_var_vec(:,1),0.25);
legend('Proportion of variance for M','Means of the proportion of variance for permuted M')
title('Bar Plot of the proportion of variance, Column 1 permuted')
xlabel('Stimulus conditions')

subplot(2,2,2)
bar(total_var_vec_M,0.5,'r')
hold on
bar(total_var_vec(:,2),0.25);
legend('Proportion of variance for M','Means of the proportion of variance for permuted M')
title('Bar Plot of the proportion of variance, Column 2 permuted')
xlabel('Stimulus conditions')

subplot(2,2,3)
bar(total_var_vec_M,0.5,'r')
hold on
bar(total_var_vec(:,3),0.25);
legend('Proportion of variance for M','Means of the proportion of variance for permuted M')
title('Bar Plot of the proportion of variance, Column 3 permuted')
xlabel('Stimulus conditions')

subplot(2,2,4)
bar(total_var_vec_M,0.5,'r')
hold on
bar(total_var_vec(:,4),0.25);
legend('Proportion of variance for M','Means of the proportion of variance for permuted M')
title('Bar Plot of the proportion of variance, Column 4 permuted')
xlabel('Stimulus conditions')
suptitle('Null distribution of PC loadings for all variables')
%%
%PART D
clear all
load hw4_data1
for iteration=1:1:500
    
    random_indices=randi(100,1,100);
    for c=1:1:4
        for i=1:1:100
            permuted_M(i,c)=M(random_indices(i),c);
        end
    end
    
    [eig_vecs,eig_vals]=pca(permuted_M);
    
    for col=1:1:4
        var_vec(col)=var(eig_vals(:,col));
    end
    var_vec=var_vec./(sum(var_vec));
    
    var_matrix(:,iteration)=var_vec;
end

for l=1:1:4
    total_var_vec(l)=mean(var_matrix(l,:));
end

[eig_vecs_M,eig_vals_M]=pca(M);

for col=1:1:4
    var_vec_M(col)=var(eig_vals_M(:,col));
end
total_var_vec_M=var_vec_M./(sum(var_vec_M));


figure(4)    
bar(total_var_vec_M,0.5,'r')
hold on
bar(total_var_vec,0.25);
hold on 
legend('Proportion of variance for M','Means of the proportion of variance for bootstrapped M')
title('Bar Plot of the proportion of variance')
xlabel('Stimulus conditions')

%%
%PART E

for iteration=1:1:500
    
    for c=1:1:4
        random_indices=randi(100,1,100);
        for i=1:1:100
            permuted_M(i,c)=M(random_indices(i),c);
        end
    end
    
    [eig_vecs,eig_vals]=pca(permuted_M);
    
    for col=1:1:4
        var_vec(col)=var(eig_vals(:,col));
    end
    var_vec=var_vec./(sum(var_vec));
    
    var_matrix(:,iteration)=var_vec;
end

for l=1:1:4
    total_var_vec(l)=mean(var_matrix(l,:));
end

[eig_vecs_M,eig_vals_M]=pca(M);

for col=1:1:4
    var_vec_M(col)=var(eig_vals_M(:,col));
end
total_var_vec_M=var_vec_M./(sum(var_vec_M));


figure(4)    
bar(total_var_vec_M,0.5,'r')
hold on
bar(total_var_vec,0.25);
hold on 
errorbar(total_var_vec,total_var_vec_M);
legend('Proportion of variance for M','Means of the proportion of variance for independently bootstrapped M')
title('Bar Plot of the proportion of variance')
xlabel('Stimulus conditions')
%%
%QUESTION 3
%PART A

load hw4_data2

%Principal Analysis

[V,D]=eig(faces'*faces);

s=0;
for i=1:1:length(D)
    s=s+D(i,i);
end

for i=0:1:99
    p_var(i+1)=D(length(D)-i,length(D)-i)/s;
end

figure(1)
plot(p_var)
title('Proportion of variance explained by each individual PC, for the first 100 PCs');

for i=0:1:24
    PC_matrix_25(:,i+1)=V(:,length(V)-i);
end

figure(2)
dispImArray(transpose(PC_matrix_25));
title('Displayed first 25 PCs');

%%
%PART B

%For 36 faces
%
PC_proj_matrix_10_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:9
        PC_proj_matrix_10_36f(:,j)=PC_proj_matrix_10_36f(:,j)+(dot(V(:,length(V)-i),faces(j,:))).*V(:,length(V)-i);
    end
end
figure(3)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(PC_proj_matrix_10_36f));
title('Displayed first 36 faces based on first 10 PCs');

%
PC_proj_matrix_25_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:24
        PC_proj_matrix_25_36f(:,j)=PC_proj_matrix_25_36f(:,j)+(dot(V(:,length(V)-i),faces(j,:))).*V(:,length(V)-i);
    end
end
figure(4)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(PC_proj_matrix_25_36f));
title('Displayed first 36 faces based on first 25 PCs');

%
PC_proj_matrix_50_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:49
        PC_proj_matrix_50_36f(:,j)=PC_proj_matrix_50_36f(:,j)+(dot(V(:,length(V)-i),faces(j,:))).*V(:,length(V)-i);
    end
end
figure(5)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(PC_proj_matrix_50_36f));
title('Displayed first 36 faces based on first 50 PCs');

%For 1000 faces
PC_proj_matrix_10_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:9
        PC_proj_matrix_10_1000f(:,j)=PC_proj_matrix_10_1000f(:,j)+(dot(V(:,length(V)-i),faces(j,:))).*V(:,length(V)-i);
    end
end

PC_proj_matrix_25_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:24
        PC_proj_matrix_25_1000f(:,j)=PC_proj_matrix_25_1000f(:,j)+(dot(V(:,length(V)-i),faces(j,:))).*V(:,length(V)-i);
    end
end

PC_proj_matrix_50_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:49
        PC_proj_matrix_50_1000f(:,j)=PC_proj_matrix_50_1000f(:,j)+(dot(V(:,length(V)-i),faces(j,:))).*V(:,length(V)-i);
    end
end


PC_proj_matrix_10_1000f=transpose(PC_proj_matrix_10_1000f);
PC_proj_matrix_25_1000f=transpose(PC_proj_matrix_25_1000f);
PC_proj_matrix_50_1000f=transpose(PC_proj_matrix_50_1000f);

for i=1:1:1000
    MSE_10(i)=mean((faces(i,:)-(PC_proj_matrix_10_1000f(i,:))).^2);
    MSE_25(i)=mean((faces(i,:)-(PC_proj_matrix_25_1000f(i,:))).^2);
    MSE_50(i)=mean((faces(i,:)-(PC_proj_matrix_50_1000f(i,:))).^2);
end

for i=1:1:36
    MSE_10_36f(i)=mean((faces(i,:)-(PC_proj_matrix_10_1000f(i,:))).^2);
    MSE_25_36f(i)=mean((faces(i,:)-(PC_proj_matrix_25_1000f(i,:))).^2);
    MSE_50_36f(i)=mean((faces(i,:)-(PC_proj_matrix_50_1000f(i,:))).^2);
end

mean_MSE_10_36f=mean(MSE_10_36f);
std_MSE_10_36f=std(MSE_10_36f);

mean_MSE_25_36f=mean(MSE_25_36f);
std_MSE_25_36f=std(MSE_25_36f);

mean_MSE_50_36f=mean(MSE_50_36f);
std_MSE_50_36f=std(MSE_50_36f);

mean_MSE_10=mean(MSE_10);
std_MSE_10=std(MSE_10);

mean_MSE_25=mean(MSE_25);
std_MSE_25=std(MSE_25);

mean_MSE_50=mean(MSE_50);
std_MSE_50=std(MSE_50);


mean_MSE_10_36f
mean_MSE_25_36f
mean_MSE_50_36f
std_MSE_10_36f
std_MSE_25_36f
std_MSE_50_36f

mean_MSE_10
mean_MSE_25
mean_MSE_50
std_MSE_10
std_MSE_25
std_MSE_50
%%
%PART C

[icasig_10] = fastica(faces, 'lastEig', 50, 'numOfIC', 10);
[icasig_25] = fastica(faces, 'lastEig', 50, 'numOfIC', 25);
[icasig_50] = fastica(faces, 'lastEig', 50, 'numOfIC', 50);

figure(6)
dispImArray(icasig_10);
title('Obtained 10 ICs');

figure(7)
dispImArray(icasig_25);
title('Obtained 25 ICs');

figure(8)
dispImArray(icasig_50);
title('Obtained 50 ICs');


IC_proj_matrix_10_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:9
        IC_proj_matrix_10_36f(:,j)=IC_proj_matrix_10_36f(:,j)+transpose((dot(icasig_10(10-i,:),faces(j,:))).*icasig_10(10-i,:));
    end
end
figure(9)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(IC_proj_matrix_10_36f));
title('Displayed first 36 faces based on first 10 ICs')

IC_proj_matrix_25_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:24
        IC_proj_matrix_25_36f(:,j)=IC_proj_matrix_25_36f(:,j)+transpose((dot(icasig_25(25-i,:),faces(j,:))).*icasig_25(25-i,:));
    end
end
figure(10)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(IC_proj_matrix_25_36f));
title('Displayed first 36 faces based on first 25 ICs')

IC_proj_matrix_50_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:9
        IC_proj_matrix_50_36f(:,j)=IC_proj_matrix_50_36f(:,j)+transpose((dot(icasig_50(50-i,:),faces(j,:))).*icasig_50(50-i,:));
    end
end
figure(11)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(IC_proj_matrix_50_36f));
title('Displayed first 36 faces based on first 50 ICs')

%%%Finding the means and stds of MSEs
%For 1000 faces
IC_proj_matrix_10_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:9
        IC_proj_matrix_10_1000f(:,j)=IC_proj_matrix_10_1000f(:,j)+transpose((dot(icasig_10(10-i,:),faces(j,:))).*icasig_10(10-i,:));
    end
end

IC_proj_matrix_25_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:24
        IC_proj_matrix_25_1000f(:,j)=IC_proj_matrix_25_1000f(:,j)+transpose((dot(icasig_25(25-i,:),faces(j,:))).*icasig_25(25-i,:));
    end
end

IC_proj_matrix_50_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:49
        IC_proj_matrix_50_1000f(:,j)=IC_proj_matrix_50_1000f(:,j)+transpose((dot(icasig_50(50-i,:),faces(j,:))).*icasig_50(50-i,:));
    end
end


IC_proj_matrix_10_1000f=transpose(IC_proj_matrix_10_1000f);
IC_proj_matrix_25_1000f=transpose(IC_proj_matrix_25_1000f);
IC_proj_matrix_50_1000f=transpose(IC_proj_matrix_50_1000f);

for i=1:1:1000
    MSE_10_IC(i)=mean((faces(i,:)-(IC_proj_matrix_10_1000f(i,:))).^2);
    MSE_25_IC(i)=mean((faces(i,:)-(IC_proj_matrix_25_1000f(i,:))).^2);
    MSE_50_IC(i)=mean((faces(i,:)-(IC_proj_matrix_50_1000f(i,:))).^2);
end

mean_MSE_10_IC=mean(MSE_10_IC);
std_MSE_10_IC=std(MSE_10_IC);

mean_MSE_25_IC=mean(MSE_25_IC);
std_MSE_25_IC=std(MSE_25_IC);

mean_MSE_50_IC=mean(MSE_50_IC);
std_MSE_50_IC=std(MSE_50_IC);

mean_MSE_10_IC
mean_MSE_25_IC
mean_MSE_50_IC
std_MSE_10_IC
std_MSE_25_IC
std_MSE_50_IC
%%
%PART D

positive_faces=faces+(-min(faces(:)));

[~,MF_10]=nnmf(positive_faces,10);
[~,MF_25]=nnmf(positive_faces,25);
[~,MF_50]=nnmf(positive_faces,50);

figure(12)
dispImArray(MF_10);
title('Displayed first 10 MFs');

figure(13)
dispImArray(MF_25);
title('Displayed first 25 MFs');

figure(14)
dispImArray(MF_50);
title('Displayed first 50 MFs');


MF_proj_matrix_10_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:9
        MF_proj_matrix_10_36f(:,j)=MF_proj_matrix_10_36f(:,j)+transpose((dot(MF_10(i+1,:),faces(j,:))).*MF_10(i+1,:));
    end
end
figure(15)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(MF_proj_matrix_10_36f));
title('Displayed first 36 faces based on first 10 MFs')

MF_proj_matrix_25_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:9
        MF_proj_matrix_25_36f(:,j)=MF_proj_matrix_25_36f(:,j)+transpose((dot(MF_25(i+1,:),faces(j,:))).*MF_25(i+1,:));
    end
end
figure(16)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(MF_proj_matrix_25_36f));
title('Displayed first 36 faces based on first 25 MFs')

MF_proj_matrix_50_36f=zeros(1024,36);
for j=1:1:36
    for i=0:1:9
        MF_proj_matrix_50_36f(:,j)=MF_proj_matrix_50_36f(:,j)+transpose((dot(MF_50(i+1,:),faces(j,:))).*MF_50(i+1,:));
    end
end
figure(17)
subplot(1,2,1)
dispImArray(faces(1:36,:))
title('Original Images')
subplot(1,2,2)
dispImArray(transpose(MF_proj_matrix_50_36f));
title('Displayed first 36 faces based on first 50 MFs')

%%%Finding the means and stds of MSEs
%For 1000 faces
MF_proj_matrix_10_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:9
        MF_proj_matrix_10_1000f(:,j)=MF_proj_matrix_10_1000f(:,j)+transpose((dot(MF_10(10-i,:),faces(j,:))).*MF_10(10-i,:));
    end
end

MF_proj_matrix_25_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:24
        MF_proj_matrix_25_1000f(:,j)=MF_proj_matrix_25_1000f(:,j)+transpose((dot(MF_25(25-i,:),faces(j,:))).*MF_25(25-i,:));
    end
end

MF_proj_matrix_50_1000f=zeros(1024,1000);
for j=1:1:1000
    for i=0:1:49
        MF_proj_matrix_50_1000f(:,j)=MF_proj_matrix_50_1000f(:,j)+transpose((dot(MF_50(50-i,:),faces(j,:))).*MF_50(50-i,:));
    end
end


MF_proj_matrix_10_1000f=transpose(MF_proj_matrix_10_1000f);
MF_proj_matrix_25_1000f=transpose(MF_proj_matrix_25_1000f);
MF_proj_matrix_50_1000f=transpose(MF_proj_matrix_50_1000f);

for i=1:1:1000
    MSE_10_MF(i)=mean((faces(i,:)-(MF_proj_matrix_10_1000f(i,:))).^2);
    MSE_25_MF(i)=mean((faces(i,:)-(MF_proj_matrix_25_1000f(i,:))).^2);
    MSE_50_MF(i)=mean((faces(i,:)-(MF_proj_matrix_50_1000f(i,:))).^2);
end

mean_MSE_10_MF=mean(MSE_10_MF);
std_MSE_10_MF=std(MSE_10_MF);

mean_MSE_25_MF=mean(MSE_25_MF);
std_MSE_25_MF=std(MSE_25_MF);

mean_MSE_50_MF=mean(MSE_50_MF);
std_MSE_50_MF=std(MSE_50_MF);

mean_MSE_10_MF
mean_MSE_25_MF
mean_MSE_50_MF
std_MSE_10_MF
std_MSE_25_MF
std_MSE_50_MF

%%
%QUESTION 4
%PART A
load hw4_data3

%USING EUCLIDEAN DISTANCE
%Finding the distance between the response i and the other responses
Euc_distances_matrix=zeros(length(stype),length(stype));
for j=1:1:length(stype);
    for i=1:1:length(stype);
        Euc_distances_matrix(j,i)=pdist2(vresp(j,:),vresp(i,:));
    end
end

figure(1)
imagesc(Euc_distances_matrix);
title('Confusion matrix for euclidean distance metric');


% Finding the means of each row of distances
Means_of_Euc_distances=zeros(181,1);
for i=1:1:181
    Means_of_Euc_distances(i)=mean(Euc_distances_matrix(i,:));
end

%Creating the Estimation Matrix
Estimation_matrix_Euc=zeros(181,181);
for r=1:1:181
    for c=1:1:181
        if Euc_distances_matrix(r,c)<Means_of_Euc_distances(r)
            Estimation_matrix_Euc(r,c)=stype(r);
        else
            if stype(r)==1
                Estimation_matrix_Euc(r,c)=2;
            else
                Estimation_matrix_Euc(r,c)=1;
            end
        end
    end
end

count1=0;
count2=0;

for c=1:1:181
    for r=1:1:181
        if Estimation_matrix_Euc(r,c)==1
            count1=count1+1;
        else
            count2=count2+1;
        end
    end
    if count1>count2
        Estimated_array_Euc(c)=1;
    else
        Estimated_array_Euc(c)=2;
    end
    count1=0;
    count2=0;
end

euclidean_error_count=length(find(Estimated_array_Euc~=stype))
% 
% %Confusion_matrix = 
% figure(1)
% imagesc(confusionmat(stype,Estimated_array_Euc));
% title('Confusion matrix for euclidean distance metric')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%USING COSINE DISTANCE

%Finding the distance between the response i and the other responses
Cos_distances_matrix=zeros(length(stype),length(stype));
for j=1:1:length(stype);
    for i=1:1:length(stype);
        Cos_distances_matrix(j,i)=pdist2(vresp(j,:),vresp(i,:),'cosine');
    end
end
figure(2)
imagesc(Cos_distances_matrix);
title('Confusion matrix for cosine distance metric')

% Finding the means of each row of distances
Means_of_Cos_distances=zeros(181,1);
for i=1:1:181
    Means_of_Cos_distances(i)=mean(Cos_distances_matrix(i,:));
end

%Creating the Estimation Matrix
Estimation_matrix_Cos=zeros(181,181);
for r=1:1:181
    for c=1:1:181
        if Cos_distances_matrix(r,c)<Means_of_Cos_distances(r)
            Estimation_matrix_Cos(r,c)=stype(r);
        else
            if stype(r)==1
                Estimation_matrix_Cos(r,c)=2;
            else
                Estimation_matrix_Cos(r,c)=1;
            end
        end
    end
end

count1=0;
count2=0;

for c=1:1:181
    for r=1:1:181
        if Estimation_matrix_Cos(r,c)==1
            count1=count1+1;
        else
            count2=count2+1;
        end
    end
    if count1>count2
        Estimated_array_Cos(c)=1;
    else
        Estimated_array_Cos(c)=2;
    end
    count1=0;
    count2=0;
end

cosine_error_count=length(find(Estimated_array_Cos~=stype))
% figure(2)
% imagesc(confusionmat(stype,Estimated_array_Cos));
% title('Confusion matrix for cosine distance metric')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%USING CORRELATION DISTANCE

%Finding the distance between the response i and the other responses
Corr_distances_matrix=zeros(length(stype),length(stype));
for j=1:1:length(stype);
    for i=1:1:length(stype);
        Corr_distances_matrix(j,i)=pdist2(vresp(j,:),vresp(i,:),'correlation');
    end
end
figure(3)
imagesc(Corr_distances_matrix);
title('Confusion matrix for correlation distance metric')

% Finding the means of each row of distances
Means_of_Corr_distances=zeros(181,1);
for i=1:1:181
    Means_of_Corr_distances(i)=mean(Corr_distances_matrix(i,:));
end

%Creating the Estimation Matrix
Estimation_matrix_Corr=zeros(181,181);
for r=1:1:181
    for c=1:1:181
        if Corr_distances_matrix(r,c)<Means_of_Corr_distances(r)
            Estimation_matrix_Corr(r,c)=stype(r);
        else
            if stype(r)==1
                Estimation_matrix_Corr(r,c)=2;
            else
                Estimation_matrix_Corr(r,c)=1;
            end
        end
    end
end

count1=0;
count2=0;

for c=1:1:181
    for r=1:1:181
        if Estimation_matrix_Corr(r,c)==1
            count1=count1+1;
        else
            count2=count2+1;
        end
    end
    if count1>count2
        Estimated_array_Corr(c)=1;
    else
        Estimated_array_Corr(c)=2;
    end
    count1=0;
    count2=0;
end

correlation_error_count=length(find(Estimated_array_Corr~=stype))
% figure(3)
% imagesc(confusionmat(stype,Estimated_array_Corr));
% title('Confusion matrix for correlation distance metric')


%%
%PART B

MDS_analysis_Euc=cmdscale(Euc_distances_matrix);
figure(4)
for i=1:1:180
    if stype(i)==1
        scatter(MDS_analysis_Euc(i,1),MDS_analysis_Euc(i,2),'b');
        hold on
    else
        scatter(MDS_analysis_Euc(i,1),MDS_analysis_Euc(i,2),'r');
    end
end
title('Scatter plot of the stimuli, Euclidean distance metric is used')
xlabel('Projections onto the first MDS component')
ylabel('Projections onto the second MDS component')


for j=1:1:181
    Cos_distances_matrix(j,j)=0;
    Corr_distances_matrix(j,j)=0;
end
MDS_analysis_Cos=cmdscale(Cos_distances_matrix);
figure(5)
for i=1:1:180
    if stype(i)==1
        scatter(MDS_analysis_Cos(i,1),MDS_analysis_Cos(i,2),'b');
        hold on
    else
        scatter(MDS_analysis_Cos(i,1),MDS_analysis_Cos(i,2),'r');
    end
end
title('Scatter plot of the stimuli, cosine distance metric is used')
xlabel('Projections onto the first MDS component')
ylabel('Projections onto the second MDS component')

MDS_analysis_Corr=cmdscale(Corr_distances_matrix);
figure(6)
for i=1:1:180
    if stype(i)==1
        scatter(MDS_analysis_Corr(i,1),MDS_analysis_Corr(i,2),'b');
        hold on
    else
        scatter(MDS_analysis_Corr(i,1),MDS_analysis_Corr(i,2),'r');
    end
end
title('Scatter plot of the stimuli, correlation distance metric is used')
xlabel('Projections onto the first MDS component')
ylabel('Projections onto the second MDS component')


%%
%QUESTION 5
%PART A
load hw4_data4

%Generating the distance matrices
Euc_dist_matrix=zeros(length(conn),length(conn));
Cos_dist_matrix=zeros(length(conn),length(conn));
Corr_dist_matrix=zeros(length(conn),length(conn));

%Finding the distance between the response i and the other responses by
%using different distance metrics
for j=1:1:length(conn);
    for i=1:1:length(conn);
        Euc_dist_matrix(j,i)=pdist2(tresp(j,:),tresp(i,:));
        Cos_dist_matrix(j,i)=pdist2(tresp(j,:),tresp(i,:),'cosine');
        Corr_dist_matrix(j,i)=pdist2(tresp(j,:),tresp(i,:),'correlation');
    end
end

%Calculating cluster solutions using different distance metrics for single
%linkage
Euc_single_link=linkage(Euc_dist_matrix,'single');
Cos_single_link=linkage(Cos_dist_matrix,'single');
Corr_single_link=linkage(Corr_dist_matrix,'single');

%Calculating cluster solutions using different distance metrics for average
%linkage
Euc_average_link=linkage(Euc_dist_matrix,'average');
Cos_average_link=linkage(Cos_dist_matrix,'average');
Corr_average_link=linkage(Corr_dist_matrix,'average');

%Calculating cluster solutions using different distance metrics for
%complete linkage
Euc_complete_link=linkage(Euc_dist_matrix,'complete');
Cos_complete_link=linkage(Cos_dist_matrix,'complete');
Corr_complete_link=linkage(Corr_dist_matrix,'complete');

%Plotting the results
figure(1)
subplot(1,3,1)
dendrogram(Euc_single_link)
title('Euclidean Distance Metric');
subplot(1,3,2)
dendrogram(Cos_single_link)
title('Cosine Distance Metric');
subplot(1,3,3)
dendrogram(Corr_single_link)
title('Correlation Distance Metric');
suptitle('Single Linkage');

figure(2)
subplot(1,3,1)
dendrogram(Euc_average_link)
title('Euclidean Distance Metric');
subplot(1,3,2)
dendrogram(Cos_average_link)
title('Cosine Distance Metric');
subplot(1,3,3)
dendrogram(Corr_average_link)
title('Correlation Distance Metric');
suptitle('Average Linkage');

figure(3)
subplot(1,3,1)
dendrogram(Euc_complete_link)
title('Euclidean Distance Metric');
subplot(1,3,2)
dendrogram(Cos_complete_link)
title('Cosine Distance Metric');
subplot(1,3,3)
dendrogram(Corr_complete_link)
title('Correlation Distance Metric');
suptitle('Complete Linkage');

%%
%PART B

%Generating the cluster indices arrays
Euc_indices_array=zeros(length(conn),1);
Cos_indices_array=zeros(length(conn),1);
Corr_indices_array=zeros(length(conn),1);

%Calculating the cluster indices of each observation using different
%distance metrics
Euc_indices_array=kmeans(tresp,3,'Distance','sqeuclidean');
Cos_indices_array=kmeans(tresp,3,'Distance','cosine');
Corr_indices_array=kmeans(tresp,3,'Distance','correlation');

%Plotting the results
figure(4)
subplot(1,3,1)
silhouette(tresp,Euc_indices_array,'Euclidean');
title('Euclidean type metric')
subplot(1,3,2)
silhouette(tresp,Cos_indices_array,'cosine');
title('Cosine type metric')
subplot(1,3,3)
silhouette(tresp,Corr_indices_array,'correlation');
title('Correlation type metric')
suptitle('Cluster solutions based on the distance metrics ? {Euclidean, cosine, correlation} ');

%%
%PART C

%Clustering the
Euc_single_indices=cluster(Euc_single_link,'maxclust',3);
Cos_single_indices=cluster(Cos_single_link,'maxclust',3);
Corr_single_indices=cluster(Corr_single_link,'maxclust',3);

Euc_average_indices=cluster(Euc_average_link,'maxclust',3);
Cos_average_indices=cluster(Cos_average_link,'maxclust',3);
Corr_average_indices=cluster(Corr_average_link,'maxclust',3);

Euc_complete_indices=cluster(Euc_complete_link,'maxclust',3);
Cos_complete_indices=cluster(Cos_complete_link,'maxclust',3);
Corr_complete_indices=cluster(Corr_complete_link,'maxclust',3);

%Calculating the randindex
result_indice_Euc_single=randindex(conn,Euc_single_indices);
result_indice_Cos_single=randindex(conn,Cos_single_indices);
result_indice_Corr_single=randindex(conn,Corr_single_indices);

result_indice_Euc_average=randindex(conn,Euc_average_indices);
result_indice_Cos_average=randindex(conn,Cos_average_indices);
result_indice_Corr_average=randindex(conn,Corr_average_indices);

result_indice_Euc_complete=randindex(conn,Euc_complete_indices);
result_indice_Cos_complete=randindex(conn,Cos_complete_indices);
result_indice_Corr_complete=randindex(conn,Corr_complete_indices);

result_indice_Euc_kmeans=randindex(conn,Euc_indices_array);
result_indice_Cos_kmeans=randindex(conn,Cos_indices_array);
result_indice_Corr_kmeans=randindex(conn,Corr_indices_array);

resulted_indices_names={'result_indice_Euc_single','result_indice_Cos_single','result_indice_Corr_single','result_indice_Euc_average','result_indice_Cos_average','result_indice_Corr_average','result_indice_Euc_kmeans','result_indice_Cos_kmeans','result_indice_Corr_kmeans','result_indice_Euc_complete','result_indice_Cos_complete','result_indice_Corr_complete'};
resulted_indices=[result_indice_Euc_single,result_indice_Cos_single,result_indice_Corr_single,result_indice_Euc_average,result_indice_Cos_average,result_indice_Corr_average,result_indice_Euc_kmeans,result_indice_Cos_kmeans,result_indice_Corr_kmeans,result_indice_Euc_complete,result_indice_Cos_complete,result_indice_Corr_complete];

[m,i]=max(resulted_indices);
m
resulted_indices_names(i)
disp('Cluster analysis works best when k-means and cosine distance metric is used');
%%
%PART D

[Cos_indices_array,centroid]=kmeans(tresp,3,'Distance','cosine');

cluster1r=1;
cluster2r=1;
cluster3r=1;

for c=1:1:4
    for r=1:1:150
        if Cos_indices_array(r)==1
            cluster1_matrix(cluster1r,c)=tresp(r,c);
            cluster1r=cluster1r+1;
        elseif Cos_indices_array(r)==2
            cluster2_matrix(cluster2r,c)=tresp(r,c);
            cluster2r=cluster2r+1;
        else
            cluster3_matrix(cluster3r,c)=tresp(r,c);
            cluster3r=cluster3r+1;
        end
    end
    cluster1r=1;
    cluster2r=1;
    cluster3r=1;
end

for cluster=1:1:3
    for c=1:1:4
        if cluster==1
            standard_deviations(cluster,c)=std(cluster1_matrix(:,c));
            means_matrix(cluster,c)=mean(cluster1_matrix(:,c));
        elseif cluster==2
            standard_deviations(cluster,c)=std(cluster2_matrix(:,c));
            means_matrix(cluster,c)=mean(cluster2_matrix(:,c));
        else
            standard_deviations(cluster,c)=std(cluster3_matrix(:,c));
            means_matrix(cluster,c)=mean(cluster3_matrix(:,c));
        end
    end
end

figure(5)
subplot(1,3,1)
bar([standard_deviations(1,:);means_matrix(1,:)]')
title('Cluster 1 Means and STDs')
legend('STD','MEAN')
subplot(1,3,2)
bar([standard_deviations(2,:);means_matrix(2,:)]')
title('Cluster 2 Means and STDs')
legend('STD','MEAN')
subplot(1,3,3)
bar([standard_deviations(3,:);means_matrix(3,:)]')
title('Cluster 3 Means and STDs')
legend('STD','MEAN')
suptitle('Mean and std of response amplitudes for each cognitive task');

%%
%QUESTION 7
%PART A
load hw4_data6
mani(M);

% %PART B
% mani(M);

%PART C
figure(1)
scatter(M(1,:),M(2,:))
title('Scatter plot of the data points');

%Running k-means algorithm
res=kmeans(transpose(M),2);

%Ploting the data using different colors for different clusters 
%according to the result of the k-means algorithm.
figure(2)
for i=1:1:1000
    if res(i)==1
        scatter(M(1,i),M(2,i),'b');
        hold on
    else
        scatter(M(1,i),M(2,i),'r');
    end
end
title('Clustering using k-means');

%Running and plotting specral clustering algorithm
figure(3)
[~,~,~]=spcl(transpose(M),2,'gaussdist',0.5);


