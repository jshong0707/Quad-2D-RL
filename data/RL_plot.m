%% MATLAB Plotting Script for training_logs.mat
% 이 스크립트는 training_logs.mat 파일을 불러와서 각 metric(필드)을 플롯합니다.
% 각 metric은 2열 배열이어야 하며, 첫 번째 열은 global timestep(steps)이고,
% 두 번째 열은 metric 값입니다.

clc;
clear;
close all;

% MAT 파일 로드
matFile = 'training_logs.mat';
if exist(matFile, 'file')
    data = load(matFile);
else
    error('파일 %s 가 존재하지 않습니다.', matFile);
end

% 로드한 파일에 포함된 필드 이름 목록
fields = fieldnames(data);

% 각 필드를 하나의 figure 창에 플롯
for i = 1:length(fields)
    fieldName = fields{i};
    metric = data.(fieldName);
    
    % metric 데이터가 2열 이상인 숫자 배열인지 확인
    if isnumeric(metric) && size(metric,2) >= 2
        % 첫 번째 열: steps, 두 번째 열: 값
        steps = metric(:, 1);
        values = metric(:, 2);
        
        figure;  % 별도의 figure 창 생성
        plot(steps, values, '-', 'LineWidth', 1.5, 'MarkerSize', 6);
        grid on;
        xlabel('Steps');
        ylabel(fieldName, 'Interpreter', 'Latex', 'FontName', 'Times new roman');
        title(sprintf('%s', fieldName), 'Interpreter', 'Latex', 'FontName', 'Times new roman');

        % 현재 figure를 PNG 파일로 저장
        cleanFieldName = regexprep(fieldName, '\W', '_');
        filename = sprintf('figure_%02d_%s.png', i, cleanFieldName);
        print(gcf, '-dpng', filename);
    else
        fprintf('필드 %s 는 2열 이상의 numeric 배열이 아닙니다.\n', fieldName);
    end
end

setFigurePositions(4);