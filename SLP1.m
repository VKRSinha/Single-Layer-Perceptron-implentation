load fisheriris.mat
meas=[randi(1,size(meas,1),1),meas];%x0=1 appended in every instance of dataset
Y=rand(size(meas,1),1);
w=rand(5,3);%weight vector initialized ..each column for a class
A=randperm(150,150)';%an array of randomly shuffled indices
meas=meas(A,:);%meas shuffled
species=species(A,:);% species shuffled in same order as that of meas
traindata=meas(1:0.7*150,:);
testdata=meas(0.7*150+1:150,:);
result=species(0.7*150+1:150,:);%for test data
for i=1:size(species,1)
    if (strcmp(species(i),'virginica'))
        Y(i)=1;
    end
    if (strcmp(species(i),'versicolor'))
        Y(i)=2;
    end
    if (strcmp(species(i),'setosa'))
        Y(i)=3;
    end
end

% SLP classification on IRIS DATA
for p=1:3
    for i=1:size(Y,1)%making class specific binary class vector
        if (Y(i)==p)
            Y(i)=1;
        else 
            Y(i)=0;
        end
    end
    for k=1:100%stochastic gradient descent
         for j=1:size(traindata,1)
            x=traindata(j,:);
            y=x*w(:,p);            
            h=(1/(1+exp(-y)));
            if (((h>0.5)&&(Y(j)==0))||(h<0.5)&&(Y(j)==1))%misclassification
                w(:,p)=w(:,p)+0.05*(Y(j)-h)*x';%update weights with (eta)=0.05
            end 
         end
    end
end
count=0;
for i=1:size(testdata,1)
    x=testdata(i,:);
    max=0;
    pm=1;
    for p=1:3
        y=x*w(:,p);
        h=1/(1+exp(-y));
        if (h>max)
            max=h;
            pm=p;
        end
    end
    if ((p==1&&(strcmp(result(i),'virginica')))||(p==2&&(strcmp(result(i),'versicolor')))||(p==3&&(strcmp(result(i),'setosa'))))
        count=count+1;
    end
end
accuracy=(count/size(testdata,1))*100

    
    
    
    
    
    
    
        

        
    
        
            
        

