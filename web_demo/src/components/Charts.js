/* eslint-disable import/no-anonymous-default-export */
import React from 'react'
import { DualAxes } from '@ant-design/charts';

export default ({ chartData, setCheckedList }) => {
    const [data, setData] = React.useState([])

    // const onPlotReady = (plot) => {
    //     plot.on('legend-item:click', ({ view }) => {
    //         console.log(view.geometries[0].data);
    //         let shownColumn = new Set(view.geometries[0].data.map(item => item.column))
    //         setCheckedList(Array.from(shownColumn))
    //         console.log(Array.from(shownColumn))
    //     });
    // };

    React.useEffect(() => {
        setData(chartData)
    }, [chartData]);

    const config = {
        data: [data, data],
        xField: 'date',
        yField: ['value', 'scores'],
        xAxis: {
            title: {
                text: 'Date'
            },
            type: 'time',
        },
        yAxis: {
            value: {
                title: {
                    text: 'Values'
                }
            },
            scores: {
                title: {
                    text: 'Anomaly Scores'
                }
            }
        },
        legend: {
            layout: 'horizontal',
            position: 'bottom'
        },
        geometryOptions: [
            {
                geometry: 'line',
                seriesField: 'column',
                smooth: true,
            },
            {
                geometry: 'column',
                columnWidthRatio: 1,
                color: 'rgba(255,0,0,0.05)',
                style: {
                    opacity: 0.2
                }
            },
        ],
        annotations: {
            scores: [
                {
                    type: 'line',
                    top: true,
                    start: ['min', 5],
                    end: ['max', 5],
                    text: {
                        content: 'Anomaly Threshold',
                        position: 'start',
                        autoRotate: false,
                        style: {
                            fill: 'red'
                        }
                    },
                    style: {
                        lineWidth: 2,
                        lineDash: [3, 3],
                        stroke: 'red'
                    }
                },
            ]
        }
    };

    return (
        <React.Fragment >
            <DualAxes {...config} />
        </React.Fragment >
    )
}