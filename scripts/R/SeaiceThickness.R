#Loading in packages 
library(ggplot2)
library(dplyr)
library(ggfortify)
library(tidyverse)
library(tidyr)
library(matrixStats)
library(DMwR)
library(directlabels)
library(reshape2)
library(lubridate)

#Reading in csv that contains SIT averages over time
dd<-read.csv("/Users/fridaperez/Desktop/SIT_Plots/sit_may_oct.csv")
dd<-as.data.frame(dd)
print(dd)

#Finding missing values
is.na(dd)

#Getting table summaries by adding new column of the average per month over the years as well as the standard deviation
sit_year<-cbind(dd,"Mean"=rowMeans(dd[,c("X2003","X2004","X2005","X2006","X2007","X2008","X2009", "X2010", "X2011")],na.rm=TRUE))
print(sit_year)
sit_year<-sit_year %>%
  mutate(stDev = apply(.[(2:10)],1,sd,na.rm=TRUE))
print(sit_year)

#Calculating the lower and upper confidence intervals and then adding it as columns
lower.ci = sit_year$Mean- qnorm(0.975)*sit_year$stDev/sqrt(9) # 9 years
upper.ci= sit_year$Mean+ qnorm(0.975)*sit_year$stDev/sqrt(9)

L.CI<-as.numeric(unlist(lower.ci))
U.CI<-as.numeric(unlist(upper.ci))

print(L.CI)
print(U.CI)

sit_year$lower_ci <- L.CI
sit_year$upper_ci <- U.CI

print(sit_year)

#Now we have to turn data into long format instead of wide so it can be read by ggplot
df<-melt(sit_year, id.vars = c("months"))
print(df)

##Getting the long format in a csv to make some final tweaks
write.csv(df,"/Users/fridaperez/Desktop/sit_ci.csv",row.names=TRUE)

##reading in new long format 
df<-read.csv("/Users/fridaperez/Desktop/sit_ci.csv")
df<-as.data.frame(df)
print(df)

##plotting it short version
plt<-ggplot(df, aes(x=df$Months, y= df$SIT, color=df$Years))+ 
  geom_line()+
  ggtitle("Sea Ice Thickness Variation") +
  labs(y="SIT (m)", x="May-October")+
  labs(colour= "Years")+
  geom_dl(aes(label = df$Years), method = list(dl.combine("first.points", "last.points"), cex = 0.01)) +
  geom_dl(aes(label = df$Years), method = list(dl.trans(x = x + 0.2), "last.points", cex = 0.8)) +
  geom_dl(aes(label = df$Years), method = list(dl.trans(x = x - 0.2), "first.points", cex = 0.8)) 
plt
## adding confidence interval ribbon 
plt2<- plt+ geom_ribbon(aes(x=df$Months,ymin=df$lower_ci,ymax=df$upper_ci), fill="steelblue", alpha=.3)
plt2

## NEW PLOT
# now we want to see SIT over the years with trend line
yearly<-read.csv("/Users/fridaperez/Desktop//SIT_Plots/time.csv")
yearly<-as.data.frame(yearly)
print(yearly)

## Plotting by years 
theme_set(theme_minimal())
# Demo dataset
head(yearly)
str(yearly)

# changing dates 


time<- c("10/15/2002","9/15/2011")
ymd(time)

# Basic line plot
ggplot(data = yearly, aes(x = yearly$Dates, y = yearly$SIT))+
  geom_point(color = "#00AFBB", size = 2)

