function plot_sr(im, cl, xlabel_, ylabel_, XTick_pos,...
                   XTick_labels,YTick_pos, YTick_labels, show)
    if show==1
        figure('visible','off');
    else
        figure();
    end
       
    imagesc(im, cl);
    xlabel(xlabel_, 'FontSize', 12);
    ylabel(ylabel_, 'FontSize', 12);
    colorbar;
    set(gca, 'XTick', XTick_pos, 'XTickLabel', XTick_labels,'fontsize',18); % 10 ticks 
    set(gca, 'YTick', YTick_pos, 'YTickLabel', YTick_labels,'fontsize',18); % 20 ticks
    colormap(customcolormap_preset('red-white-blue'));
    axis('square')
end