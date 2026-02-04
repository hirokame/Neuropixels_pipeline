function ExcelPrint(prefix,filename,T,folder)

mydir=cd;
if isdir([mydir '/Data'])==0
mkdir('Data')
else
end

newname=strjoin([prefix,'_',filename,'.csv']);
if newname{1}(1)=="_"
    newname=string(newname{1}(2:end));
end
% a= [char(folder),'\Data'];
a= [char(folder),'/Data']; % for mac
mydir=cd;
cd(a)

writetable(T,newname,'WriteVariableNames',false);

cd (mydir)
end

